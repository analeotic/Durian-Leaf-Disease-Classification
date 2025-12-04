# Data Summary - Durian Leaf Disease Classification

## Dataset Overview

### Class Distribution

| Class | Description | ภาษาไทย | Count | Percentage |
|-------|-------------|---------|-------|------------|
| 0 | Healthy | ไม่เป็นโรค | 41 | 14.7% |
| 1 | Worms & Beetles | หนอนและแมงปีกแข็ง | 78 | 28.1% |
| 2 | Fungal Diseases | เชื้อรา | 87 | 31.3% |
| 3 | Aphids & Mites | เพลี้ย | 72 | 25.9% |
| **Total** | | | **278** | **100%** |

### Class 2: Fungal Diseases (เชื้อรา) Details
- ใบจุด (Leaf Spot)
- ใบจุดสาหร่าย (Algal Leaf Spot)
- ราสนิม (Rust)
- ใบไหม้ (Leaf Blight)
- ฟิวซาเรี่ยม (Fusarium)
- รากเน่าโคนเน่า (Root and Stem Rot)
- ราดำ (Sooty Mold)

### Class 3: Aphids & Mites (เพลี้ย) Details
- จั๊กจั่นฝอย (Leafhopper)
- เพลี้ยไก่แจ้ (Whitefly)
- เพลี้ยนาสาร (Thrips)
- เพลี้ยแป้ง (Mealybug)
- เพลี้ยหอย (Scale Insect)
- ไรแดง (Red Spider Mite)

## Dataset Files

### Training Data
- **File**: `data/raw/train.csv`
- **Samples**: 278 images
- **Columns**:
  - `id`: Image filename
  - `label`: Class label (0-3)
- **Purpose**: Training and validation split

### Test Data
- **File**: `data/raw/test.csv`
- **Samples**: 278 images
- **Columns**:
  - `id`: Image filename
- **Purpose**: Final evaluation and submission

### Submission Template
- **File**: `data/raw/submit.csv`
- **Columns**:
  - `id`: Image filename
  - `predict`: Predicted class label (to be filled)

## Class Imbalance Analysis

### Imbalance Ratio
- Most frequent class (Class 2): 87 samples
- Least frequent class (Class 0): 41 samples
- **Imbalance ratio**: 2.12:1

### Recommendations

1. **Data Augmentation**
   - Aggressive augmentation for Class 0 (minority class)
   - Standard augmentation for other classes
   - Implemented: Flips, rotations, color jitter, blur, coarse dropout

2. **Loss Function**
   - Using Label Smoothing (0.1) to prevent overconfidence
   - Alternative: Class-weighted loss

3. **Validation Strategy**
   - Stratified train-test split (80:20)
   - Ensures balanced class distribution in validation

4. **Metrics**
   - Accuracy (overall performance)
   - F1-Score (weighted) - accounts for class imbalance
   - Confusion Matrix - per-class performance

## Image Characteristics

- **Format**: JPG images
- **Expected size**: Variable (will be resized to 224x224 for model input)
- **Channels**: RGB (3 channels)
- **Content**: Close-up photos of durian leaves showing various disease symptoms

## Preprocessing Pipeline

1. **Resize**: All images resized to 224×224
2. **Normalization**: ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
3. **Data Augmentation** (Training only):
   - Random horizontal flip (p=0.5)
   - Random vertical flip (p=0.5)
   - Random 90° rotation (p=0.5)
   - Shift, scale, rotate (p=0.5)
   - Color jittering (p=0.5)
   - Blur (p=0.3)
   - Coarse dropout (p=0.3)

## Expected Performance

Based on dataset size and class distribution:

- **Baseline accuracy**: ~31% (random guessing weighted by class distribution)
- **Target accuracy**: >85% with ConvNeXt + augmentation
- **Expected F1-Score**: >0.80 (weighted)

## Notes

- Small dataset (278 samples per split) requires:
  - Pretrained models (using ImageNet weights)
  - Heavy data augmentation
  - Careful monitoring for overfitting
  - Early stopping mechanism

- Class imbalance is moderate, manageable with proper techniques

- Test set same size as train set indicates potential for good generalization if trained carefully
