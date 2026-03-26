"""
Inference script for EfficientSegNet with real-time processing and robot integration.
Includes point cloud processing, visualization, and robotic manipulation examples.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
import json
import logging
from dataclasses import dataclass

from efficient_segnet_model import EfficientSegNet

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    instance_labels: np.ndarray  # (N,) instance IDs
    confidence: np.ndarray  # (N,) per-point confidence
    epistemic_uncertainty: np.ndarray  # (N,) model uncertainty
    aleatoric_uncertainty: np.ndarray  # (N,) data uncertainty
    embeddings: np.ndarray  # (N, D) learned embeddings
    inference_time: float  # seconds
    num_instances: int  # number of detected instances


class SegmentationInference:
    """Inference engine for point cloud segmentation."""
    
    def __init__(self,
                 model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_mc_samples: int = 5):
        """
        Initialize inference engine.
        
        Args:
            model_path: path to trained model checkpoint
            device: 'cuda' or 'cpu'
            num_mc_samples: number of MC dropout samples for uncertainty
        """
        self.device = torch.device(device)
        self.num_mc_samples = num_mc_samples
        
        # Load model
        self.model = EfficientSegNet(
            in_channels=3,
            feature_dim=512,
            hidden_dim=256,
            num_mc_samples=num_mc_samples,
            max_instances=32,
        ).to(self.device)
        
        # Load checkpoint if exists
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model path {model_path} not found, using untrained model")
        
        self.model.eval()
    
    def process_point_cloud(self,
                           points: np.ndarray,
                           return_uncertainty: bool = True,
                           confidence_threshold: float = 0.5,
                           max_points: int = 10000) -> SegmentationResult:
        """
        Segment a point cloud.
        
        Args:
            points: (N, 3) point cloud coordinates
            return_uncertainty: whether to compute uncertainty estimates
            confidence_threshold: filter predictions below this confidence
            max_points: downsample if more points
        
        Returns:
            SegmentationResult with segmentation and uncertainty
        """
        
        # Preprocess
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        # Convert to tensor
        points_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(
                points_tensor,
                return_uncertainty=return_uncertainty,
                mc_samples=self.num_mc_samples if return_uncertainty else 1
            )
        inference_time = time.time() - start_time
        
        # Extract results
        instance_labels = outputs['instance_labels'][0].cpu().numpy()
        confidence = outputs['confidence'][0].cpu().numpy()
        epistemic = outputs['epistemic_uncertainty'][0].cpu().numpy()
        aleatoric = outputs['aleatoric_uncertainty'][0].cpu().numpy()
        embeddings = outputs['embeddings'][0].cpu().numpy()
        
        # Apply confidence filtering
        low_confidence_mask = confidence < confidence_threshold
        instance_labels[low_confidence_mask] = 0  # Mark as background
        
        # Count instances
        num_instances = len(np.unique(instance_labels)) - 1  # Exclude background
        
        return SegmentationResult(
            instance_labels=instance_labels,
            confidence=confidence,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            embeddings=embeddings,
            inference_time=inference_time,
            num_instances=num_instances,
        )
    
    def batch_process(self,
                     point_clouds: List[np.ndarray],
                     return_uncertainty: bool = True) -> List[SegmentationResult]:
        """
        Process multiple point clouds in batch.
        
        Args:
            point_clouds: list of (N_i, 3) point clouds
            return_uncertainty: whether to compute uncertainties
        
        Returns:
            list of SegmentationResult
        """
        results = []
        for points in point_clouds:
            result = self.process_point_cloud(points, return_uncertainty)
            results.append(result)
        return results


class PostProcessor:
    """Post-process segmentation results."""
    
    @staticmethod
    def extract_instances(result: SegmentationResult,
                         points: np.ndarray,
                         min_points: int = 50) -> Dict[int, np.ndarray]:
        """
        Extract individual instances from segmentation.
        
        Args:
            result: segmentation result
            points: (N, 3) original point cloud
            min_points: minimum points per instance
        
        Returns:
            dict mapping instance ID to (N_i, 3) point cloud
        """
        instances = {}
        
        for inst_id in np.unique(result.instance_labels):
            if inst_id == 0:  # Skip background
                continue
            
            mask = result.instance_labels == inst_id
            instance_points = points[mask]
            
            if len(instance_points) >= min_points:
                instances[int(inst_id)] = instance_points
        
        return instances
    
    @staticmethod
    def compute_instance_properties(instance_points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute properties of an instance.
        
        Args:
            instance_points: (N, 3) point cloud of single instance
        
        Returns:
            dict with center, bounds, covariance, etc.
        """
        center = instance_points.mean(axis=0)
        bounds = np.array([instance_points.min(axis=0), instance_points.max(axis=0)])
        extent = bounds[1] - bounds[0]
        
        # Covariance for principal component analysis
        centered = instance_points - center
        covariance = np.cov(centered.T)
        
        # Eigenvalues/eigenvectors for orientation
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        sorted_indices = np.argsort(-eigenvalues)
        principal_axes = eigenvectors[:, sorted_indices]
        
        return {
            'center': center,
            'bounds': bounds,
            'extent': extent,
            'volume': np.prod(extent),
            'covariance': covariance,
            'principal_axes': principal_axes,
            'eigenvalues': eigenvalues[sorted_indices],
        }
    
    @staticmethod
    def refine_boundaries(result: SegmentationResult,
                         points: np.ndarray,
                         refinement_threshold: float = 0.3) -> np.ndarray:
        """
        Refine instance boundaries using uncertainty.
        Points with high epistemic uncertainty at boundaries are refined.
        
        Args:
            result: segmentation result
            points: (N, 3) original points
            refinement_threshold: uncertainty threshold for refinement
        
        Returns:
            refined instance labels (N,)
        """
        refined_labels = result.instance_labels.copy()
        
        # Identify boundary points (high epistemic uncertainty)
        boundary_mask = result.epistemic_uncertainty > refinement_threshold
        
        if boundary_mask.sum() == 0:
            return refined_labels
        
        boundary_indices = np.where(boundary_mask)[0]
        boundary_points = points[boundary_indices]
        
        # For each boundary point, find nearest confident point
        for idx in boundary_indices:
            if result.instance_labels[idx] != 0:
                continue  # Already assigned
            
            # Find k-nearest neighbors among confident points
            confident_mask = result.confidence > 0.7
            confident_indices = np.where(confident_mask)[0]
            
            if len(confident_indices) > 0:
                distances = np.linalg.norm(points[confident_indices] - points[idx], axis=1)
                nearest_idx = confident_indices[distances.argmin()]
                
                # Assign to nearest confident instance
                refined_labels[idx] = refined_labels[nearest_idx]
        
        return refined_labels


class RobotGraspPlanner:
    """Plan grasps for robot manipulation based on segmentation."""
    
    def __init__(self, gripper_width: float = 0.1, max_grasp_quality: float = 1.0):
        """
        Initialize grasp planner.
        
        Args:
            gripper_width: maximum gripper opening width (meters)
            max_grasp_quality: maximum quality score
        """
        self.gripper_width = gripper_width
        self.max_grasp_quality = max_grasp_quality
    
    def plan_grasps(self,
                   instance_id: int,
                   instance_points: np.ndarray,
                   confidence: float,
                   num_candidates: int = 10) -> List[Dict]:
        """
        Plan grasps for a segmented instance.
        
        Args:
            instance_id: instance identifier
            instance_points: (N, 3) point cloud of instance
            confidence: confidence score of segmentation
            num_candidates: number of grasp candidates
        
        Returns:
            list of grasp dicts with pose and quality
        """
        
        if len(instance_points) < 10:
            return []
        
        # Compute instance properties
        properties = PostProcessor.compute_instance_properties(instance_points)
        center = properties['center']
        principal_axes = properties['principal_axes']
        extent = properties['extent']
        
        grasps = []
        
        # Generate grasp candidates around instance
        for i in range(num_candidates):
            # Random approach direction
            approach_angle = 2 * np.pi * i / num_candidates
            approach_dir = np.array([
                np.cos(approach_angle),
                np.sin(approach_angle),
                0.5,  # Downward bias
            ])
            approach_dir = approach_dir / np.linalg.norm(approach_dir)
            
            # Grasp center (slightly above instance center)
            grasp_center = center + np.array([0, 0, extent[2] / 2])
            
            # Gripper width based on instance extent
            required_width = min(extent[0], extent[1])
            gripper_quality = 1.0 - min(1.0, required_width / self.gripper_width)
            
            # Combine with segmentation confidence
            grasp_quality = confidence * gripper_quality
            
            grasps.append({
                'instance_id': instance_id,
                'center': grasp_center,
                'approach_direction': approach_dir,
                'quality': grasp_quality,
                'gripper_width': required_width,
                'confidence': confidence,
            })
        
        # Sort by quality
        grasps = sorted(grasps, key=lambda g: g['quality'], reverse=True)
        
        return grasps
    
    def select_best_grasp(self, grasps: List[Dict], quality_threshold: float = 0.5) -> Optional[Dict]:
        """
        Select best grasp from candidates.
        
        Args:
            grasps: list of grasp dicts
            quality_threshold: minimum quality threshold
        
        Returns:
            best grasp dict or None
        """
        if not grasps:
            return None
        
        for grasp in grasps:
            if grasp['quality'] >= quality_threshold:
                return grasp
        
        return None


class PerformanceMonitor:
    """Monitor inference performance metrics."""
    
    def __init__(self):
        self.inference_times = []
        self.confidence_scores = []
        self.uncertainty_values = []
    
    def record_inference(self, result: SegmentationResult):
        """Record inference metrics."""
        self.inference_times.append(result.inference_time)
        self.confidence_scores.extend(result.confidence)
        self.uncertainty_values.extend(
            result.epistemic_uncertainty + result.aleatoric_uncertainty
        )
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'mean_inference_time': np.mean(self.inference_times),
            'median_inference_time': np.median(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'mean_confidence': np.mean(self.confidence_scores),
            'std_confidence': np.std(self.confidence_scores),
            'mean_uncertainty': np.mean(self.uncertainty_values),
            'std_uncertainty': np.std(self.uncertainty_values),
            'throughput_fps': 1.0 / np.mean(self.inference_times),
        }
    
    def print_stats(self):
        """Print performance statistics."""
        stats = self.get_stats()
        print("\n=== Performance Statistics ===")
        for key, value in stats.items():
            if 'time' in key or 'throughput' in key:
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value:.4f}")


# Example usage and robot integration
def example_robot_manipulation():
    """Example robot manipulation workflow."""
    
    logger.info("Initializing segmentation inference...")
    segmenter = SegmentationInference('model_checkpoint.pt', device='cuda')
    
    logger.info("Loading sample point cloud...")
    # In practice, this would come from a real RGB-D sensor
    points = np.random.randn(5000, 3) * 0.5  # Synthetic data
    
    logger.info("Running segmentation...")
    result = segmenter.process_point_cloud(
        points,
        return_uncertainty=True,
        confidence_threshold=0.5,
    )
    
    logger.info(f"Segmentation completed in {result.inference_time:.3f}s")
    logger.info(f"Detected {result.num_instances} instances")
    
    # Post-process
    logger.info("Extracting instances...")
    instances = PostProcessor.extract_instances(result, points)
    logger.info(f"Extracted {len(instances)} valid instances")
    
    # Plan grasps
    logger.info("Planning grasps...")
    grasp_planner = RobotGraspPlanner()
    all_grasps = []
    
    for inst_id, inst_points in instances.items():
        instance_confidence = result.confidence[result.instance_labels == inst_id].mean()
        
        if instance_confidence < 0.5:
            logger.info(f"Skipping instance {inst_id} (low confidence: {instance_confidence:.3f})")
            continue
        
        grasps = grasp_planner.plan_grasps(inst_id, inst_points, instance_confidence)
        all_grasps.extend(grasps)
        
        logger.info(f"Planned {len(grasps)} grasps for instance {inst_id}")
    
    # Select best grasp
    if all_grasps:
        best_grasp = grasp_planner.select_best_grasp(all_grasps, quality_threshold=0.6)
        if best_grasp:
            logger.info(
                f"Selected grasp for instance {best_grasp['instance_id']} "
                f"(quality: {best_grasp['quality']:.3f})"
            )
            logger.info(f"Grasp center: {best_grasp['center']}")
            logger.info(f"Required gripper width: {best_grasp['gripper_width']:.3f}m")
    
    return result, instances, all_grasps


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    result, instances, grasps = example_robot_manipulation()
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    monitor.record_inference(result)
    monitor.print_stats()
