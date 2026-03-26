"""
Utility functions for EfficientSegNet:
- Point cloud visualization
- Evaluation metrics
- Data augmentation
- Uncertainty visualization
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PointCloudVisualizer:
    """Visualize point clouds with segmentation and uncertainty."""
    
    @staticmethod
    def visualize_segmentation(points: np.ndarray,
                              instance_labels: np.ndarray,
                              confidence: Optional[np.ndarray] = None,
                              title: str = "Instance Segmentation",
                              figsize: Tuple[int, int] = (12, 5)):
        """
        Visualize segmented point cloud.
        
        Args:
            points: (N, 3) point coordinates
            instance_labels: (N,) instance IDs
            confidence: (N,) optional confidence scores
            title: plot title
            figsize: figure size
        """
        fig = plt.figure(figsize=figsize)
        
        # Colored by instance
        ax1 = fig.add_subplot(121, projection='3d')
        colors = cm.tab20(np.linspace(0, 1, len(np.unique(instance_labels))))
        
        for inst_id in np.unique(instance_labels):
            mask = instance_labels == inst_id
            color = colors[inst_id % len(colors)]
            
            ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                       c=[color], s=10, alpha=0.6)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f"{title} - By Instance")
        
        # Colored by confidence (if provided)
        if confidence is not None:
            ax2 = fig.add_subplot(122, projection='3d')
            scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                                 c=confidence, cmap='RdYlGn', s=10, alpha=0.6)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title(f"{title} - By Confidence")
            plt.colorbar(scatter, ax=ax2, label='Confidence')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_uncertainty(points: np.ndarray,
                             epistemic: np.ndarray,
                             aleatoric: np.ndarray,
                             combined: Optional[np.ndarray] = None,
                             figsize: Tuple[int, int] = (15, 4)):
        """
        Visualize different uncertainty types.
        
        Args:
            points: (N, 3) point coordinates
            epistemic: (N,) epistemic uncertainty
            aleatoric: (N,) aleatoric uncertainty
            combined: (N,) combined uncertainty (optional)
            figsize: figure size
        """
        num_plots = 3 if combined is None else 4
        fig = plt.figure(figsize=figsize)
        
        # Epistemic uncertainty
        ax1 = fig.add_subplot(1, num_plots, 1, projection='3d')
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                              c=epistemic, cmap='hot', s=10, alpha=0.6)
        ax1.set_title('Epistemic Uncertainty')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter1, ax=ax1)
        
        # Aleatoric uncertainty
        ax2 = fig.add_subplot(1, num_plots, 2, projection='3d')
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                              c=aleatoric, cmap='hot', s=10, alpha=0.6)
        ax2.set_title('Aleatoric Uncertainty')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.colorbar(scatter2, ax=ax2)
        
        # Combined uncertainty
        if combined is None:
            combined = epistemic + aleatoric
        
        ax3 = fig.add_subplot(1, num_plots, 3, projection='3d')
        scatter3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                              c=combined, cmap='hot', s=10, alpha=0.6)
        ax3.set_title('Combined Uncertainty')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        plt.colorbar(scatter3, ax=ax3)
        
        # Confidence (inverse of combined)
        if num_plots == 4:
            confidence = np.exp(-combined)
            ax4 = fig.add_subplot(1, num_plots, 4, projection='3d')
            scatter4 = ax4.scatter(points[:, 0], points[:, 1], points[:, 2],
                                  c=confidence, cmap='RdYlGn', s=10, alpha=0.6)
            ax4.set_title('Confidence')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z')
            plt.colorbar(scatter4, ax=ax4)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_instance_error(points: np.ndarray,
                                predictions: np.ndarray,
                                targets: np.ndarray,
                                confidence: np.ndarray,
                                figsize: Tuple[int, int] = (12, 5)):
        """
        Visualize prediction errors.
        
        Args:
            points: (N, 3) point coordinates
            predictions: (N,) predicted labels
            targets: (N,) ground truth labels
            confidence: (N,) confidence scores
            figsize: figure size
        """
        fig = plt.figure(figsize=figsize)
        
        # Error map (correct=green, incorrect=red)
        is_correct = predictions == targets
        colors_error = np.zeros((len(points), 3))
        colors_error[is_correct, 1] = 1  # Green for correct
        colors_error[~is_correct, 0] = 1  # Red for incorrect
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors_error, s=10, alpha=0.6)
        ax1.set_title('Predictions (Green=Correct, Red=Error)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Error-confidence relationship
        ax2 = fig.add_subplot(122)
        ax2.scatter(confidence[is_correct], np.ones(is_correct.sum()),
                   color='green', alpha=0.5, label='Correct')
        ax2.scatter(confidence[~is_correct], np.zeros((~is_correct).sum()),
                   color='red', alpha=0.5, label='Incorrect')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Correctness')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Incorrect', 'Correct'])
        ax2.legend()
        ax2.set_title('Confidence vs Correctness')
        
        plt.tight_layout()
        return fig


class EvaluationMetrics:
    """Comprehensive evaluation metrics for instance segmentation."""
    
    @staticmethod
    def compute_all_metrics(predictions: np.ndarray,
                           targets: np.ndarray,
                           confidence: np.ndarray,
                           max_instances: int = 32) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: (N,) predicted instance labels
            targets: (N,) ground truth labels
            confidence: (N,) confidence scores
            max_instances: maximum instance ID
        
        Returns:
            dict of metric names and values
        """
        
        # Filter valid points
        valid_mask = targets >= 0
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        confidence = confidence[valid_mask]
        
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = EvaluationMetrics._accuracy(predictions, targets)
        metrics['mAP'] = EvaluationMetrics._mean_ap(predictions, targets, confidence)
        metrics['mIoU'] = EvaluationMetrics._mean_iou(predictions, targets, max_instances)
        
        # Instance metrics
        metrics['mPQ'], metrics['mSQ'], metrics['mRQ'] = \
            EvaluationMetrics._panoptic_metrics(predictions, targets)
        
        # Uncertainty metrics
        metrics['AUC_ROC'] = EvaluationMetrics._auroc(predictions, targets, confidence)
        metrics['ECE'] = EvaluationMetrics._calibration_error(predictions, targets, confidence)
        
        return metrics
    
    @staticmethod
    def _accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Accuracy: fraction of correctly predicted points."""
        return float((predictions == targets).mean())
    
    @staticmethod
    def _mean_ap(predictions: np.ndarray, targets: np.ndarray,
                confidence: np.ndarray, iou_threshold: float = 0.5) -> float:
        """Mean Average Precision at IoU threshold."""
        
        unique_targets = np.unique(targets)
        unique_targets = unique_targets[unique_targets >= 0]
        
        if len(unique_targets) == 0:
            return 0.0
        
        # Sort by confidence
        sorted_indices = np.argsort(-confidence)
        
        tp_array = np.zeros(len(sorted_indices))
        fp_array = np.zeros(len(sorted_indices))
        
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
                tp_array[i] = 1
                matched_targets.add(best_target)
            else:
                fp_array[i] = 1
        
        tp_cumsum = np.cumsum(tp_array)
        fp_cumsum = np.cumsum(fp_array)
        
        recalls = tp_cumsum / len(unique_targets)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return float(ap)
    
    @staticmethod
    def _mean_iou(predictions: np.ndarray, targets: np.ndarray,
                 num_classes: int = 32) -> float:
        """Mean Intersection over Union."""
        ious = []
        
        for class_id in range(1, num_classes + 1):
            pred_mask = predictions == class_id
            target_mask = targets == class_id
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
        
        return float(np.mean(ious)) if ious else 0.0
    
    @staticmethod
    def _panoptic_metrics(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float, float]:
        """Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ)."""
        
        unique_targets = np.unique(targets)
        unique_targets = unique_targets[unique_targets >= 0]
        
        tp = 0
        fp = 0
        fn = 0
        iou_sum = 0
        
        matched_preds = set()
        
        for target_id in unique_targets:
            target_mask = targets == target_id
            
            best_iou = 0
            best_pred = None
            
            for pred_id in np.unique(predictions):
                if pred_id == 0 or pred_id in matched_preds:
                    continue
                
                pred_mask = predictions == pred_id
                intersection = np.logical_and(pred_mask, target_mask).sum()
                union = np.logical_or(pred_mask, target_mask).sum()
                
                iou = intersection / union if union > 0 else 0.0
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_id
            
            if best_iou > 0.5 and best_pred is not None:
                tp += 1
                iou_sum += best_iou
                matched_preds.add(best_pred)
            else:
                fn += 1
        
        # False positives
        for pred_id in np.unique(predictions):
            if pred_id > 0 and pred_id not in matched_preds:
                fp += 1
        
        sq = iou_sum / tp if tp > 0 else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
        pq = sq * rq
        
        return float(pq), float(sq), float(rq)
    
    @staticmethod
    def _auroc(predictions: np.ndarray, targets: np.ndarray,
              confidence: np.ndarray) -> float:
        """Area Under ROC curve for correct vs incorrect predictions."""
        
        is_correct = (predictions == targets).astype(int)
        
        sorted_indices = np.argsort(-confidence)
        is_correct_sorted = is_correct[sorted_indices]
        
        tp_array = np.cumsum(is_correct_sorted)
        fp_array = np.cumsum(1 - is_correct_sorted)
        
        tpr = tp_array / (is_correct.sum() + 1e-8)
        fpr = fp_array / ((1 - is_correct).sum() + 1e-8)
        
        # Trapezoidal rule for AUC
        auc = np.trapz(tpr, fpr)
        
        return float(auc)
    
    @staticmethod
    def _calibration_error(predictions: np.ndarray, targets: np.ndarray,
                          confidence: np.ndarray, num_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)."""
        
        is_correct = (predictions == targets).astype(float)
        
        bin_edges = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
            
            if mask.sum() == 0:
                continue
            
            bin_confidence = confidence[mask].mean()
            bin_accuracy = is_correct[mask].mean()
            
            ece += mask.sum() / len(confidence) * abs(bin_confidence - bin_accuracy)
        
        return float(ece)


class DataAugmentation:
    """Data augmentation for point clouds."""
    
    @staticmethod
    def random_rotation(points: np.ndarray) -> np.ndarray:
        """Apply random rotation around Z-axis."""
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        rotation = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ])
        
        return points @ rotation.T
    
    @staticmethod
    def random_scaling(points: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random isotropic scaling."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return points * scale
    
    @staticmethod
    def random_jitter(points: np.ndarray, jitter_std: float = 0.01) -> np.ndarray:
        """Apply Gaussian noise."""
        return points + np.random.normal(0, jitter_std, points.shape)
    
    @staticmethod
    def random_translation(points: np.ndarray, translation_range: float = 0.2) -> np.ndarray:
        """Apply random translation."""
        translation = np.random.uniform(-translation_range, translation_range, 3)
        return points + translation
    
    @staticmethod
    def augment(points: np.ndarray, labels: np.ndarray,
               augmentation_types: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentations.
        
        Args:
            points: (N, 3) point cloud
            labels: (N,) instance labels
            augmentation_types: list of augmentation types to apply
        
        Returns:
            augmented points and labels
        """
        
        if augmentation_types is None:
            augmentation_types = ['rotation', 'scaling', 'jitter', 'translation']
        
        augmented_points = points.copy()
        
        for aug_type in augmentation_types:
            if aug_type == 'rotation':
                augmented_points = DataAugmentation.random_rotation(augmented_points)
            elif aug_type == 'scaling':
                augmented_points = DataAugmentation.random_scaling(augmented_points)
            elif aug_type == 'jitter':
                augmented_points = DataAugmentation.random_jitter(augmented_points)
            elif aug_type == 'translation':
                augmented_points = DataAugmentation.random_translation(augmented_points)
        
        return augmented_points, labels


class ConfigManager:
    """Manage configuration files."""
    
    @staticmethod
    def save_config(config: Dict, path: str):
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {path}")
    
    @staticmethod
    def load_config(path: str) -> Dict:
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {path}")
        return config
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default configuration."""
        return {
            'model': {
                'in_channels': 3,
                'feature_dim': 512,
                'hidden_dim': 256,
                'max_instances': 32,
                'num_mc_samples': 5,
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'num_epochs': 100,
                'num_points': 4096,
            },
            'inference': {
                'confidence_threshold': 0.5,
                'num_mc_samples': 5,
                'max_points': 10000,
            },
            'augmentation': {
                'augmentation_types': ['rotation', 'scaling', 'jitter', 'translation'],
            },
        }


if __name__ == '__main__':
    # Example usage
    print("Point Cloud Segmentation Utilities")
    print("===================================")
    
    # Generate synthetic point cloud
    points = np.random.randn(5000, 3)
    instance_labels = np.random.randint(0, 10, 5000)
    confidence = np.random.uniform(0.3, 1.0, 5000)
    
    # Test visualization
    print("Creating visualization...")
    # fig = PointCloudVisualizer.visualize_segmentation(points, instance_labels, confidence)
    # plt.savefig('segmentation.png')
    # print("Saved segmentation visualization")
    
    # Test metrics
    print("Computing metrics...")
    metrics = EvaluationMetrics.compute_all_metrics(instance_labels, instance_labels, confidence)
    print(f"Metrics: {metrics}")
    
    # Test data augmentation
    print("Testing augmentation...")
    aug_points, _ = DataAugmentation.augment(points, instance_labels)
    print(f"Original shape: {points.shape}, Augmented shape: {aug_points.shape}")
