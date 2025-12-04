import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_model
from src.dataset import DurianLeafDataset, get_valid_transforms


def predict_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda'
) -> tuple:
    """Make predictions on test set"""
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(img_ids)

    return all_ids, all_preds


def main():
    parser = argparse.ArgumentParser(description='Inference on test set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, default='data/raw/test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--image_dir', type=str, default='data/images',
                        help='Directory containing images')
    parser.add_argument('--output', type=str, default='data/raw/submission.csv',
                        help='Output CSV file path')
    parser.add_argument('--model_name', type=str, default='convnext_tiny',
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of classes')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--validate', action='store_true',
                        help='If true, compute metrics (requires labels in CSV)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        num_classes=args.num_classes,
        device=device
    )
    print('Model loaded successfully')

    # Load test data
    print(f'Loading test data from {args.test_csv}...')
    test_df = pd.read_csv(args.test_csv)
    print(f'Test samples: {len(test_df)}')

    # Create test dataset and dataloader
    test_dataset = DurianLeafDataset(
        image_dir=args.image_dir,
        df=test_df,
        transform=get_valid_transforms(args.image_size),
        is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Make predictions
    print('Making predictions...')
    img_ids, predictions = predict_test_set(model, test_loader, device)

    # Create submission file
    submission_df = pd.DataFrame({
        'id': img_ids,
        'predict': predictions
    })

    # Save submission
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    submission_df.to_csv(args.output, index=False)
    print(f'Submission saved to {args.output}')

    # Show prediction distribution
    print('\nPrediction distribution:')
    pred_counts = submission_df['predict'].value_counts().sort_index()
    for class_id, count in pred_counts.items():
        print(f'  Class {class_id}: {count} samples ({count/len(submission_df)*100:.1f}%)')

    # If validation mode, compute metrics
    if args.validate and 'label' in test_df.columns:
        print('\n' + '='*60)
        print('VALIDATION METRICS')
        print('='*60)

        # Get true labels
        true_labels = test_df['label'].values

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        f1_macro = f1_score(true_labels, predictions, average='macro')

        print(f'\nAccuracy: {accuracy:.4f}')
        print(f'F1 Score (Weighted): {f1_weighted:.4f}')
        print(f'F1 Score (Macro): {f1_macro:.4f}')

        print('\nClassification Report:')
        print(classification_report(true_labels, predictions, target_names=[
            'Class 0: Healthy',
            'Class 1: Worms & Beetles',
            'Class 2: Fungal',
            'Class 3: Aphids & Mites'
        ]))
        print('='*60)


if __name__ == '__main__':
    main()
