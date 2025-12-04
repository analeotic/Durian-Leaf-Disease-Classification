import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import create_model, LabelSmoothingCrossEntropy
from dataset import create_dataloaders


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""

    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """Trainer class for ConvNeXt model"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        valid_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        device: str = 'cuda',
        use_amp: bool = True,
        checkpoint_dir: str = 'models'
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.scaler = GradScaler() if use_amp else None

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.train_losses = []
        self.valid_losses = []
        self.valid_accuracies = []
        self.valid_f1_scores = []

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.valid_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(self.valid_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return epoch_loss, accuracy, f1, all_preds, all_labels

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f'Saved best model to {path}')

        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)

    def train(self, num_epochs: int, early_stopping: EarlyStopping = None):
        """Train the model"""
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 60)

            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1, _, _ = self.validate()

            self.train_losses.append(train_loss)
            self.valid_losses.append(val_loss)
            self.valid_accuracies.append(val_acc)
            self.valid_f1_scores.append(val_f1)

            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if early_stopping:
                if early_stopping(val_loss, epoch):
                    print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                    print(f'Best epoch was {early_stopping.best_epoch + 1}')
                    break

        self.plot_training_history()

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.valid_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.valid_accuracies, label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(self.valid_f1_scores, label='Val F1 Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Validation F1 Score')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_history.png'))
        plt.close()


def main():
    """Main training function"""

    # Configuration
    CONFIG = {
        'image_dir': 'data/images',
        'csv_file': 'durian_leaf.csv',
        'model_name': 'convnext_tiny',
        'num_classes': 4,
        'image_size': 224,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_amp': True,
        'early_stopping_patience': 10,
        'checkpoint_dir': 'models',
        'test_size': 0.2,
        'random_state': 42,
        'label_smoothing': 0.1,
        'drop_rate': 0.2,
        'drop_path_rate': 0.1,
    }

    print('Configuration:')
    for key, value in CONFIG.items():
        print(f'  {key}: {value}')
    print()

    # Load and prepare data
    print('Loading data...')
    df = pd.read_csv(CONFIG['csv_file'])
    df = df[df['predict'].notna()]

    train_df, valid_df = train_test_split(
        df,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=df['predict']
    )

    print(f'Train samples: {len(train_df)}, Valid samples: {len(valid_df)}')
    print(f'Class distribution in train: {train_df["predict"].value_counts().to_dict()}')

    # Create dataloaders
    train_loader, valid_loader = create_dataloaders(
        train_df=train_df,
        valid_df=valid_df,
        image_dir=CONFIG['image_dir'],
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size'],
        num_workers=CONFIG['num_workers']
    )

    # Create model
    print('\nCreating model...')
    model = create_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        pretrained=True,
        drop_rate=CONFIG['drop_rate'],
        drop_path_rate=CONFIG['drop_path_rate']
    )
    model = model.to(CONFIG['device'])
    print(f'Model: {CONFIG["model_name"]}')
    print(f'Device: {CONFIG["device"]}')

    # Loss, optimizer, scheduler
    criterion = LabelSmoothingCrossEntropy(smoothing=CONFIG['label_smoothing'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs'],
        eta_min=1e-6
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG['early_stopping_patience'],
        mode='min'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=CONFIG['device'],
        use_amp=CONFIG['use_amp'],
        checkpoint_dir=CONFIG['checkpoint_dir']
    )

    # Train
    print('\nStarting training...')
    trainer.train(num_epochs=CONFIG['num_epochs'], early_stopping=early_stopping)

    # Final validation
    print('\nFinal validation on best model...')
    best_model_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    val_loss, val_acc, val_f1, preds, labels = trainer.validate()
    print(f'Best Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

    print('\nClassification Report:')
    print(classification_report(labels, preds))


if __name__ == '__main__':
    main()
