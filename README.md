# Durian Leaf Disease Classification

โปรเจกต์ Deep Learning สำหรับจำแนกโรคใบทุเรียน 4 ประเภท โดยใช้ ConvNeXt architecture

## Dataset Description

ข้อมูลประกอบด้วย 4 คลาส:

- **Class 0: ไม่เป็นโรค (Healthy)** - 41 รูป (14.7%)
- **Class 1: หนอนและแมงปีกแข็ง (Worms & Beetles)** - 78 รูป (28.1%)
- **Class 2: เชื้อรา (Fungal Diseases)** - 87 รูป (31.3%)
  - ใบจุด, ใบจุดสาหร่าย, ราสนิม, ใบไหม้, ฟิวซาเรี่ยม, รากเน่าโคนเน่า, ราดำ
- **Class 3: เพลี้ย (Aphids & Mites)** - 72 รูป (25.9%)
  - จั๊กจั่นฝอย, เพลี้ยไก่แจ้, เพลี้ยนาสาร, เพลี้ยแป้ง, เพลี้ยหอย, ไรแดง

**ข้อมูล:**
- Training set: 278 รูป
- Test set: 278 รูป

## Project Structure

```
durian_leaf_disease_classification/
├── src/                      # Source code
│   ├── model.py             # ConvNeXt model definition
│   ├── dataset.py           # Dataset and data loaders
│   ├── train.py             # Training script
│   └── inference.py         # Inference script for test set
├── data/                    # Data directory
│   ├── images/             # Image files (train + test)
│   └── raw/                # Raw data
│       ├── train.csv       # Training labels
│       ├── test.csv        # Test image IDs
│       └── submit.csv      # Submission template
├── checkpoints/            # Saved model checkpoints
├── notebooks/              # Jupyter notebooks
│   ├── eda.ipynb           # Exploratory data analysis
│   └── test.ipynb          # Testing notebook
├── logs/                   # Training logs
├── configs/                # Configuration files
├── scripts/                # Utility scripts
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Features

- **Model**: ConvNeXt (Tiny variant) with pretrained ImageNet weights
- **Data Augmentation**: Comprehensive augmentation using Albumentations
  - Random flips, rotations, scaling
  - Color jittering, brightness/contrast adjustments
  - Gaussian/median blur, coarse dropout
- **Training Features**:
  - Label smoothing cross-entropy loss
  - AdamW optimizer with cosine annealing LR
  - Mixed precision training (AMP)
  - Early stopping
  - Model checkpointing

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Place all images (both train and test) in `data/images/`
2. Ensure CSV files are in `data/raw/`:
   - `train.csv` - Training data with labels
   - `test.csv` - Test data (only image IDs)
   - `submit.csv` - Submission template

**train.csv format:**
```csv
id,label
image1.jpg,0
image2.jpg,1
...
```

**test.csv format:**
```csv
id
test_image1.jpg
test_image2.jpg
...
```

## Usage

### Training

Run training from the project root directory:

```bash
python src/train.py
```

Configuration parameters in `train.py`:
- `model_name`: ConvNeXt variant (default: 'convnext_tiny')
- `num_classes`: Number of disease classes (default: 4)
- `image_size`: Input image size (default: 224)
- `batch_size`: Batch size (default: 32)
- `num_epochs`: Maximum training epochs (default: 50)
- `learning_rate`: Initial learning rate (default: 1e-4)
- `early_stopping_patience`: Patience for early stopping (default: 10)

### Inference (Prediction on Test Set)

Run inference on test set:

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --test_csv data/raw/test.csv \
    --image_dir data/images \
    --output data/raw/submission.csv
```

Optional parameters:
- `--model_name`: Model architecture (default: 'convnext_tiny')
- `--batch_size`: Batch size for inference (default: 32)
- `--image_size`: Input image size (default: 224)
- `--validate`: Compute metrics if labels are available in CSV

**Validation mode** (if you have labels in test CSV):
```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --test_csv data/raw/train.csv \
    --validate
```
This will show Accuracy, F1 Score (Weighted), F1 Score (Macro), and Classification Report.

### Model Loading (Python)

```python
from src.model import load_model

model = load_model(
    checkpoint_path='checkpoints/best_model.pth',
    model_name='convnext_tiny',
    num_classes=4,
    device='cuda'
)
```

### Exploratory Data Analysis

Run the EDA notebook to analyze dataset:

```bash
jupyter notebook notebooks/eda.ipynb
```

## Model Architecture

- **Base**: ConvNeXt Tiny (28M parameters)
- **Pretrained**: ImageNet-1K
- **Regularization**:
  - Dropout: 0.2
  - Drop Path (Stochastic Depth): 0.1
  - Label Smoothing: 0.1

## Training Results

After training, you'll find:
- `checkpoints/best_model.pth`: Best model based on validation loss
- `checkpoints/checkpoint_epoch_*.pth`: Epoch checkpoints
- `checkpoints/training_history.png`: Training curves (4 plots)
  - Loss (train & validation)
  - Accuracy
  - F1 Score (Weighted)
  - F1 Score (Macro)

## Evaluation Metrics

The model is evaluated using multiple metrics:

1. **Accuracy**: Overall classification accuracy
2. **F1 Score (Weighted)**: F1 score weighted by support (number of samples per class)
   - Better for representing overall performance
3. **F1 Score (Macro)**: F1 score averaged equally across all classes
   - **More important for imbalanced datasets**
   - Treats each class equally regardless of sample size
   - Better indicator of model performance on minority classes
4. **Classification Report**: Per-class precision, recall, and F1 scores

**Why Macro F1 is Important:**
- Dataset has class imbalance (Class 0: 41 samples vs Class 2: 87 samples)
- Macro F1 ensures the model performs well on ALL classes, including minorities
- Competition/production systems often care about minority class performance

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

See [requirements.txt](requirements.txt) for full dependencies.

## License

MIT License

## Author

Thanakorn Sin-on

## Acknowledgments

- ConvNeXt: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- timm library: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
