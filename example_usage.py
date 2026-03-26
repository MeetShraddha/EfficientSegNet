"""
Complete usage examples for EfficientSegNet.
Demonstrates:
1. Model training
2. Inference and uncertainty quantification
3. Evaluation metrics
4. Robot manipulation pipeline
5. Visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Import EfficientSegNet components
from efficient_segnet_model import EfficientSegNet, EfficientSegNetLoss
from train_efficient_segnet import Trainer, S3DISDataset, PointCloudCollator
from inference_efficient_segnet import SegmentationInference, PostProcessor, RobotGraspPlanner
from utils_efficient_segnet import (
    PointCloudVisualizer, 
    EvaluationMetrics, 
    DataAugmentation,
    ConfigManager,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_inference():
    """
    Example 1: Basic inference on a single point cloud.
    This is the simplest way to use EfficientSegNet for segmentation.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Inference")
    print("="*60)
    
    # Initialize inference engine
    # In practice, load a trained model checkpoint
    logger.info("Initializing segmentation inference...")
    segmenter = SegmentationInference(
        model_path='model_checkpoint.pt',  # Use empty if model doesn't exist
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_mc_samples=5,
    )
    
    # Create synthetic point cloud (replace with real sensor data)
    logger.info("Loading point cloud...")
    points = np.random.randn(5000, 3) * 2.0
    
    # Run segmentation with uncertainty quantification
    logger.info("Running segmentation...")
    result = segmenter.process_point_cloud(
        points,
        return_uncertainty=True,
        confidence_threshold=0.5,
    )
    
    # Print results
    logger.info(f"\nSegmentation Results:")
    logger.info(f"  - Inference time: {result.inference_time*1000:.1f}ms")
    logger.info(f"  - Detected instances: {result.num_instances}")
    logger.info(f"  - Mean confidence: {result.confidence.mean():.3f}")
    logger.info(f"  - Mean epistemic uncertainty: {result.epistemic_uncertainty.mean():.4f}")
    logger.info(f"  - Mean aleatoric uncertainty: {result.aleatoric_uncertainty.mean():.4f}")
    
    return result, points


def example_2_uncertainty_analysis(result, points):
    """
    Example 2: Analyze and visualize uncertainty estimates.
    Shows how to use uncertainty for decision-making.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Uncertainty Analysis")
    print("="*60)
    
    logger.info("Analyzing uncertainty estimates...")
    
    # Separate confident vs uncertain predictions
    high_confidence_mask = result.confidence > 0.7
    low_confidence_mask = result.confidence < 0.5
    
    logger.info(f"High confidence points: {high_confidence_mask.sum()}")
    logger.info(f"Low confidence points: {low_confidence_mask.sum()}")
    
    # Analyze epistemic vs aleatoric
    logger.info("\nUncertainty breakdown:")
    logger.info(f"  - Epistemic (model uncertainty):")
    logger.info(f"    Mean: {result.epistemic_uncertainty.mean():.4f}")
    logger.info(f"    Std:  {result.epistemic_uncertainty.std():.4f}")
    logger.info(f"  - Aleatoric (data uncertainty):")
    logger.info(f"    Mean: {result.aleatoric_uncertainty.mean():.4f}")
    logger.info(f"    Std:  {result.aleatoric_uncertainty.std():.4f}")
    
    # Visualize uncertainty (requires matplotlib)
    logger.info("Creating visualization (this requires matplotlib display)...")
    try:
        fig = PointCloudVisualizer.visualize_uncertainty(
            points,
            result.epistemic_uncertainty,
            result.aleatoric_uncertainty,
        )
        # Uncomment to show/save:
        # plt.show()
        # plt.savefig('uncertainty_visualization.png')
        logger.info("Visualization created (see matplotlib output)")
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")


def example_3_instance_extraction(result, points):
    """
    Example 3: Extract individual instances from segmentation.
    Useful for separate processing of each object.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Instance Extraction")
    print("="*60)
    
    logger.info("Extracting instances...")
    instances = PostProcessor.extract_instances(
        result, 
        points,
        min_points=50,  # Minimum points per instance
    )
    
    logger.info(f"Extracted {len(instances)} valid instances\n")
    
    # Analyze each instance
    for inst_id, inst_points in instances.items():
        properties = PostProcessor.compute_instance_properties(inst_points)
        
        # Get confidence for this instance
        inst_mask = result.instance_labels == inst_id
        inst_confidence = result.confidence[inst_mask].mean()
        
        logger.info(f"Instance {inst_id}:")
        logger.info(f"  - Points: {len(inst_points)}")
        logger.info(f"  - Confidence: {inst_confidence:.3f}")
        logger.info(f"  - Center: {properties['center']}")
        logger.info(f"  - Volume: {properties['volume']:.4f}")
    
    return instances


def example_4_robot_manipulation(result, points, instances):
    """
    Example 4: Plan grasps for robot manipulation.
    Shows integration with robotic pipeline.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Robot Manipulation Planning")
    print("="*60)
    
    # Initialize grasp planner
    grasp_planner = RobotGraspPlanner(
        gripper_width=0.1,  # 10cm gripper
        max_grasp_quality=1.0,
    )
    
    logger.info("Planning grasps for detected instances...\n")
    
    all_grasps = []
    
    for inst_id, inst_points in instances.items():
        # Get confidence for this instance
        inst_mask = result.instance_labels == inst_id
        inst_confidence = result.confidence[inst_mask].mean()
        
        logger.info(f"Instance {inst_id} (confidence: {inst_confidence:.3f}):")
        
        # Skip low-confidence instances
        if inst_confidence < 0.5:
            logger.info("  → SKIPPED (low confidence)")
            continue
        
        # Plan grasps
        grasps = grasp_planner.plan_grasps(
            inst_id,
            inst_points,
            inst_confidence,
            num_candidates=5,
        )
        
        all_grasps.extend(grasps)
        
        # Show top grasp for this instance
        if grasps:
            best = grasps[0]
            logger.info(f"  → Best grasp quality: {best['quality']:.3f}")
            logger.info(f"  → Required gripper width: {best['gripper_width']:.3f}m")
    
    # Select overall best grasp
    if all_grasps:
        logger.info(f"\nTotal grasps planned: {len(all_grasps)}")
        
        best_grasp = grasp_planner.select_best_grasp(
            all_grasps,
            quality_threshold=0.6,
        )
        
        if best_grasp:
            logger.info(f"\nBEST GRASP SELECTED:")
            logger.info(f"  - Instance ID: {best_grasp['instance_id']}")
            logger.info(f"  - Quality: {best_grasp['quality']:.3f}")
            logger.info(f"  - Center: {best_grasp['center']}")
            logger.info(f"  - Can execute: ✓ PROCEED WITH EXECUTION")
        else:
            logger.info("No grasp met quality threshold - REQUEST HUMAN INTERVENTION")
    else:
        logger.info("No valid grasps planned")
    
    return all_grasps


def example_5_evaluation_metrics():
    """
    Example 5: Compute comprehensive evaluation metrics.
    Shows how to evaluate on ground truth data.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Evaluation Metrics")
    print("="*60)
    
    # Generate synthetic predictions and ground truth
    logger.info("Generating synthetic test data...")
    N = 5000
    predictions = np.random.randint(0, 20, N)
    targets = np.random.randint(0, 20, N)
    confidence = np.random.uniform(0.3, 1.0, N)
    
    # Add some correct predictions
    correct_idx = np.random.choice(N, N // 2, replace=False)
    predictions[correct_idx] = targets[correct_idx]
    confidence[correct_idx] *= 1.2  # Higher confidence for correct ones
    confidence = np.clip(confidence, 0, 1)
    
    logger.info("Computing metrics...")
    metrics = EvaluationMetrics.compute_all_metrics(
        predictions,
        targets,
        confidence,
        max_instances=20,
    )
    
    logger.info("\nEvaluation Metrics:")
    logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - mAP (AP@0.5): {metrics['mAP']:.4f}")
    logger.info(f"  - mIoU: {metrics['mIoU']:.4f}")
    logger.info(f"  - Panoptic Quality (PQ): {metrics['mPQ']:.4f}")
    logger.info(f"  - Segmentation Quality (SQ): {metrics['mSQ']:.4f}")
    logger.info(f"  - Recognition Quality (RQ): {metrics['mRQ']:.4f}")
    logger.info(f"\nUncertainty Metrics:")
    logger.info(f"  - ROC-AUC: {metrics['AUC_ROC']:.4f}")
    logger.info(f"  - Calibration Error (ECE): {metrics['ECE']:.4f}")
    
    return metrics


def example_6_data_augmentation():
    """
    Example 6: Apply data augmentation to point clouds.
    Shows how to augment training data.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Data Augmentation")
    print("="*60)
    
    logger.info("Creating synthetic data...")
    points = np.random.randn(1000, 3)
    labels = np.random.randint(0, 10, 1000)
    
    logger.info(f"Original points shape: {points.shape}")
    logger.info(f"Original points range: [{points.min():.2f}, {points.max():.2f}]")
    
    # Apply augmentations
    augmentations = ['rotation', 'scaling', 'jitter', 'translation']
    aug_points, aug_labels = DataAugmentation.augment(
        points,
        labels,
        augmentation_types=augmentations,
    )
    
    logger.info(f"\nAfter augmentation:")
    logger.info(f"  - Points shape: {aug_points.shape}")
    logger.info(f"  - Points range: [{aug_points.min():.2f}, {aug_points.max():.2f}]")
    logger.info(f"  - Labels unchanged: {np.array_equal(labels, aug_labels)}")
    
    # Individual augmentations
    logger.info(f"\nIndividual augmentations:")
    
    # Rotation
    rot_points = DataAugmentation.random_rotation(points)
    logger.info(f"  - Rotation: points rotated around Z-axis")
    
    # Scaling
    scale_points = DataAugmentation.random_scaling(points)
    logger.info(f"  - Scaling: points scaled by 0.8-1.2")
    
    # Jitter
    jitter_points = DataAugmentation.random_jitter(points)
    logger.info(f"  - Jitter: Gaussian noise added")
    
    # Translation
    trans_points = DataAugmentation.random_translation(points)
    logger.info(f"  - Translation: points shifted randomly")


def example_7_configuration_management():
    """
    Example 7: Manage configurations for reproducibility.
    Shows how to save/load experiment configurations.
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Configuration Management")
    print("="*60)
    
    # Get default configuration
    logger.info("Loading default configuration...")
    config = ConfigManager.get_default_config()
    
    logger.info("\nDefault Configuration:")
    for section, params in config.items():
        logger.info(f"  [{section}]")
        for key, value in params.items():
            logger.info(f"    {key}: {value}")
    
    # Modify configuration
    logger.info("\nModifying configuration...")
    config['training']['batch_size'] = 16
    config['training']['num_epochs'] = 200
    config['inference']['confidence_threshold'] = 0.7
    
    # Save configuration
    config_path = 'experiments/config.json'
    Path('experiments').mkdir(exist_ok=True)
    ConfigManager.save_config(config, config_path)
    logger.info(f"Saved configuration to {config_path}")
    
    # Load configuration
    loaded_config = ConfigManager.load_config(config_path)
    logger.info("Loaded configuration successfully")
    
    return config


def example_8_batch_processing():
    """
    Example 8: Process multiple point clouds efficiently.
    Shows batch processing for high throughput.
    """
    print("\n" + "="*60)
    print("EXAMPLE 8: Batch Processing")
    print("="*60)
    
    # Initialize inference engine
    logger.info("Initializing inference engine...")
    segmenter = SegmentationInference(
        model_path='model_checkpoint.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Create batch of point clouds
    logger.info("Creating batch of point clouds...")
    batch_size = 5
    point_clouds = [
        np.random.randn(np.random.randint(3000, 5000), 3) 
        for _ in range(batch_size)
    ]
    
    logger.info(f"Processing {batch_size} point clouds...")
    
    # Process batch
    results = segmenter.batch_process(
        point_clouds,
        return_uncertainty=False,  # Faster without MC dropout
    )
    
    logger.info("\nBatch Processing Results:")
    total_time = 0
    for i, result in enumerate(results):
        logger.info(f"  Cloud {i+1}: {result.num_instances} instances, "
                   f"{result.inference_time*1000:.1f}ms")
        total_time += result.inference_time
    
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Average time per cloud: {total_time/batch_size*1000:.1f}ms")
    logger.info(f"  Throughput: {batch_size/total_time:.1f} clouds/sec")


def example_9_model_architecture():
    """
    Example 9: Explore model architecture and parameters.
    Shows model structure and parameter counts.
    """
    print("\n" + "="*60)
    print("EXAMPLE 9: Model Architecture")
    print("="*60)
    
    # Create model
    logger.info("Creating EfficientSegNet model...")
    model = EfficientSegNet(
        in_channels=3,
        feature_dim=512,
        hidden_dim=256,
        downsample_ratio=0.2,
        num_mc_samples=5,
        max_instances=32,
    )
    
    # Print model summary
    logger.info("\nModel Architecture:")
    logger.info(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("\nParameter Statistics:")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Parameter ratio: {trainable_params/total_params*100:.1f}%")
    
    # Test forward pass
    logger.info("\nTesting forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        # Deterministic inference
        points = torch.randn(2, 1024, 3).to(device)
        outputs = model(points, return_uncertainty=False)
        
        logger.info(f"  Input shape: {points.shape}")
        logger.info(f"  Output instance labels: {outputs['instance_labels'].shape}")
        logger.info(f"  Output confidence: {outputs['confidence'].shape}")


def main():
    """Run all examples."""
    
    print("\n" + "="*60)
    print("EfficientSegNet: Complete Usage Examples")
    print("="*60)
    
    # Example 1: Basic inference
    result, points = example_1_basic_inference()
    
    # Example 2: Uncertainty analysis
    example_2_uncertainty_analysis(result, points)
    
    # Example 3: Instance extraction
    instances = example_3_instance_extraction(result, points)
    
    # Example 4: Robot manipulation
    grasps = example_4_robot_manipulation(result, points, instances)
    
    # Example 5: Evaluation metrics
    metrics = example_5_evaluation_metrics()
    
    # Example 6: Data augmentation
    example_6_data_augmentation()
    
    # Example 7: Configuration management
    config = example_7_configuration_management()
    
    # Example 8: Batch processing
    example_8_batch_processing()
    
    # Example 9: Model architecture
    example_9_model_architecture()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
