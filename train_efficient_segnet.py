"""
Training script for EfficientSegNet point cloud segmentation.
Includes data loading, training loop, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
import pickle

from efficient_segnet_model import EfficientSegNet, EfficientSegNetLoss


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DISDataset(Dataset):
    """
    Dataset loader for S3DIS indoor scene dataset.
    Can be adapted for ScanNet or other point cloud datasets.
    """
    def __init__(self, 
                 data_dir: str = 'data/s3dis',
                 split: str = 'train',
                 num_points: int = 4096,
                 use_color: bool = False):
        """
        Args:
            data_dir: path to dataset directory
            split: 'train' or 'val'
            num_points: number of points to sample from each scene
            use_color: whether to use RGB color as features
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.use_color = use_color
        
        # For demonstration, create synthetic data
        # In practice, load actual S3DIS/ScanNet data
        self.data_list = self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Create synthetic point cloud data for demonstration."""
        data_list = []
        num_scenes = 100 if self.split == 'train' else 20
        
        for i in range(num_scenes):
            # Simulate indoor scene with multiple objects
            num_points = self.num_points
            
            # Generate random points
            points = np.random.uniform(-5, 5, (num_points, 3))
            
            # Create instance labels (simulate 5-15 objects per scene)
            num_instances = np.random.randint(5, 15)
            instance_labels = np.zeros(num_points, dtype=np.int32)
            
            for inst_id in range(1, num_instances + 1):
                # Cluster points around random centers
                center = np.random.uniform(-5, 5, 3)
                distances = np.linalg.norm(points - center, axis=1)
                
                # Soft assignment with spatial extent
                radius = np.random.uniform(0.5, 2.0)
                mask = distances < radius
                
                instance_labels[mask] = inst_id
            
            data_list.append({
                'points': points.astype(np.float32),
                'instance_labels': instance_labels,
                'scene_id': i,
            })
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        points = torch.from_numpy(data['points'])
        labels = torch.from_numpy(data['instance_labels']).long()
        
        return {
            'points': points,
            'labels': labels,
            'scene_id': data['scene_id'],
        }


class PointCloudCollator:
    """Custom collate function for point cloud batches."""
    
    def __init__(self, max_instances: int = 32):
        self.max_instances = max_instances
    
    def __call__(self, batch):
        points_list = []
        labels_list = []
        scene_ids = []
        
        max_points = max(b['points'].shape[0] for b in batch)
        
        for sample in batch:
            points = sample['points']
            labels = sample['labels']
            
            # Pad or sample to consistent size
            if points.shape[0] < max_points:
                # Zero-pad
                pad_size = max_points - points.shape[0]
                points = torch.cat([
                    points,
                    torch.zeros(pad_size, 3)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_size,), -1, dtype=labels.dtype)
                ])
            elif points.shape[0] > max_points:
                # Random sample
                indices = torch.randperm(points.shape[0])[:max_points]
                points = points[indices]
                labels = labels[indices]
            
            points_list.append(points)
            labels_list.append(labels)
            scene_ids.append(sample['scene_id'])
        
        # Stack into batch
        points_batch = torch.stack(points_list, dim=0)  # (B, N, 3)
        labels_batch = torch.stack(labels_list, dim=0)  # (B, N)
        
        return {
            'points': points_batch,
            'labels': labels_batch,
            'scene_ids': scene_ids,
        }


class SegmentationMetrics:
    """Compute instance segmentation metrics."""
    
    @staticmethod
    def compute_iou(predictions: np.ndarray, targets: np.ndarray, 
                    num_classes: int = 32) -> float:
        """
        Compute mean Intersection over Union (IoU).
        
        Args:
            predictions: (N,) predicted labels
            targets: (N,) ground truth labels
            num_classes: number of classes
        
        Returns:
            mean_iou: mean IoU across classes
        """
        ious = []
        
        for class_id in range(1, num_classes + 1):
            pred_mask = predictions == class_id
            target_mask = targets == class_id
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
        
        return np.mean(ious) if ious else 0.0
    
    @staticmethod
    def compute_panoptic_quality(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).
        
        Args:
            predictions: (N,) predicted labels
            targets: (N,) ground truth labels
        
        Returns:
            pq, sq, rq: panoptic, segmentation, and recognition quality
        """
        unique_targets = np.unique(targets)
        unique_targets = unique_targets[unique_targets != -1]  # Ignore background
        
        tp = 0
        fp = 0
        fn = 0
        iou_sum = 0
        
        for target_id in unique_targets:
            target_mask = targets == target_id
            
            # Find best matching prediction
            unique_preds = np.unique(predictions[target_mask])
            
            if len(unique_preds) == 0:
                fn += 1
                continue
            
            # Most common prediction in target region
            pred_id = np.bincount(predictions[target_mask]).argmax()
            pred_mask = predictions == pred_id
            
            # Compute IoU
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if union == 0:
                fn += 1
            else:
                iou = intersection / union
                
                if iou > 0.5:
                    tp += 1
                    iou_sum += iou
                else:
                    fn += 1
        
        # False positives (predictions not matching any target)
        unique_preds = np.unique(predictions)
        for pred_id in unique_preds:
            pred_mask = predictions == pred_id
            if np.logical_and(pred_mask, targets >= 0).sum() == 0:
                fp += 1
        
        sq = iou_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
        pq = sq * rq
        
        return pq, sq, rq
    
    @staticmethod
    def compute_ap(predictions: np.ndarray, targets: np.ndarray, 
                   confidences: np.ndarray, iou_threshold: float = 0.5) -> float:
        """
        Compute Average Precision (AP) at specific IoU threshold.
        
        Args:
            predictions: (N,) predicted instance labels
            targets: (N,) ground truth instance labels
            confidences: (N,) prediction confidence scores
            iou_threshold: IoU threshold for positive match
        
        Returns:
            ap: average precision score
        """
        # Get unique instances
        unique_targets = np.unique(targets)
        unique_targets = unique_targets[unique_targets != -1]
        
        num_targets = len(unique_targets)
        if num_targets == 0:
            return 0.0
        
        # Sort by confidence
        sorted_indices = np.argsort(-confidences)
        
        tp = np.zeros(len(sorted_indices))
        fp = np.zeros(len(sorted_indices))
        
        matched_targets = set()
        
        for i, idx in enumerate(sorted_indices):
            pred_id = predictions[idx]
            pred_mask = predictions == pred_id
            
            best_iou = 0
            best_target = None
            
            for target_id in unique_targets:
                if target_id in matched_targets:
                    continue
                
                target_mask = targets == target_id
                intersection = np.logical_and(pred_mask, target_mask).sum()
                union = np.logical_or(pred_mask, target_mask).sum()
                
                iou = intersection / union if union > 0 else 0.0
                
                if iou > best_iou:
                    best_iou = iou
                    best_target = target_id
            
            if best_iou >= iou_threshold and best_target is not None:
                tp[i] = 1
                matched_targets.add(best_target)
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_targets
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return ap


class Trainer:
    """Training loop for EfficientSegNet."""
    
    def __init__(self,
                 model: EfficientSegNet,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 num_epochs: int = 100):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.loss_fn = EfficientSegNetLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_ap': [],
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            points = batch['points'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(points, return_uncertainty=True, mc_samples=3)
            
            # Loss computation
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for batch in pbar:
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass (single pass for efficiency)
                outputs = self.model(points, return_uncertainty=False)
                
                # Loss computation
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                # Collect predictions
                predictions = outputs['instance_labels'].cpu().numpy()
                targets = labels.cpu().numpy()
                confidence = outputs['confidence'].cpu().numpy()
                
                all_predictions.extend(predictions.flatten())
                all_targets.extend(targets.flatten())
                all_confidences.extend(confidence.flatten())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        
        # Filter out background/invalid points
        valid_mask = all_targets >= 0
        all_predictions = all_predictions[valid_mask]
        all_targets = all_targets[valid_mask]
        all_confidences = all_confidences[valid_mask]
        
        # Compute metrics
        iou = SegmentationMetrics.compute_iou(all_predictions, all_targets)
        ap = SegmentationMetrics.compute_ap(all_predictions, all_targets, all_confidences)
        pq, sq, rq = SegmentationMetrics.compute_panoptic_quality(all_predictions, all_targets)
        
        return {
            'loss': avg_loss,
            'iou': iou,
            'ap': ap,
            'pq': pq,
            'sq': sq,
            'rq': rq,
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Run training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.metrics_history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['val_iou'].append(val_metrics['iou'])
            self.metrics_history['val_ap'].append(val_metrics['ap'])
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"Val AP: {val_metrics['ap']:.4f} | "
                f"Val PQ: {val_metrics['pq']:.4f}"
            )
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                logger.info(f"Saved best model with loss {self.best_val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
        
        return self.metrics_history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics_history = checkpoint['metrics_history']
        logger.info(f"Loaded checkpoint from {path}")


def main():
    """Main training script."""
    
    # Configuration
    config = {
        'data_dir': 'data/s3dis',
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'num_points': 4096,
        'feature_dim': 512,
        'hidden_dim': 256,
        'max_instances': 32,
        'num_mc_samples': 5,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = S3DISDataset(
        data_dir=config['data_dir'],
        split='train',
        num_points=config['num_points'],
    )
    val_dataset = S3DISDataset(
        data_dir=config['data_dir'],
        split='val',
        num_points=config['num_points'],
    )
    
    # Create data loaders
    collator = PointCloudCollator(max_instances=config['max_instances'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = EfficientSegNet(
        in_channels=3,
        feature_dim=config['feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_mc_samples=config['num_mc_samples'],
        max_instances=config['max_instances'],
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
    )
    
    # Train
    metrics_history = trainer.train()
    
    # Save final model
    trainer.save_checkpoint('final_model.pt')
    
    # Save metrics
    with open('metrics_history.json', 'w') as f:
        json.dump({k: v for k, v in metrics_history.items()}, f)
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
