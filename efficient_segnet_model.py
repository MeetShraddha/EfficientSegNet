"""
EfficientSegNet: Real-Time Uncertainty-Aware Point Cloud Segmentation
Implementation of hierarchical coarse-to-fine architecture with Bayesian uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
import math


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout layer with learned dropout rate.
    Reference: Gal et al., 2017
    """
    def __init__(self, weight_regularizer: float = 1e-6, dropout_regularizer: float = 1e-5, init_min: float = 0.1, init_max: float = 0.5):
        super(ConcreteDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        # Learned dropout rate (in log space for numerical stability)
        init_p = np.random.uniform(init_min, init_max)
        self.p_logit = nn.Parameter(torch.tensor(np.log(init_p / (1 - init_p))))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.p_logit)
        return F.dropout(x, p=p.item(), training=self.training)
    
    def regularization(self) -> torch.Tensor:
        """KL divergence term for dropout regularization"""
        p = torch.sigmoid(self.p_logit)
        return self.dropout_regularizer * (-p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8))


class PointNetFeatureExtractor(nn.Module):
    """
    PointNet-style feature extraction layer.
    Learns per-point features and global features from point clouds.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(PointNetFeatureExtractor, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) point cloud features
        Returns:
            features: (B, N, out_channels)
        """
        B, N, C = x.shape
        return self.mlp(x.reshape(-1, C)).reshape(B, N, -1)


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder using multiple resolution levels.
    Similar to PointNet++ but optimized for efficiency.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 256, downsample_ratio: float = 0.2):
        super(HierarchicalEncoder, self).__init__()
        
        self.downsample_ratio = downsample_ratio
        
        # Feature extraction at multiple scales
        self.local_feat = PointNetFeatureExtractor(in_channels, 64)
        self.feat_64 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.feat_128 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Global features
        self.global_feat = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_channels),
        )
        
        # Coarse confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Concrete dropout
        self.concrete_dropout = ConcreteDropout(init_min=0.2, init_max=0.4)
    
    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: (B, N, 3) raw point coordinates
        Returns:
            features: (B, N, out_channels) per-point features
            confidence: (B, N, 1) coarse confidence scores
        """
        B, N, C = points.shape
        
        # Local feature extraction
        local_features = self.local_feat(points)  # (B, N, 64)
        
        # Reshape for MLP processing
        local_features_flat = local_features.reshape(B * N, 64)
        feat_128_flat = self.feat_64(local_features_flat)
        feat_256_flat = self.feat_128(feat_128_flat)
        
        features = feat_256_flat.reshape(B, N, 256)
        
        # Dropout for uncertainty
        features = self.concrete_dropout(features)
        
        # Global pooling
        global_features = features.max(dim=1)[0]  # (B, 256)
        global_features = self.global_feat(global_features)  # (B, 256)
        
        # Broadcast global features to all points
        global_feat_expanded = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, N, 256)
        
        # Combine local and global features
        combined_features = torch.cat([features, global_feat_expanded], dim=-1)  # (B, N, 512)
        
        # Coarse confidence
        combined_features_flat = combined_features.reshape(B * N, -1)
        coarse_confidence = self.confidence_head(combined_features_flat).reshape(B, N, 1)
        
        return combined_features, coarse_confidence


class RegionGrowingModule(nn.Module):
    """
    Learnable region growing module using GRU-based refinement.
    Predicts which points to add/remove from regions.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, num_iterations: int = 6):
        super(RegionGrowingModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_iterations = num_iterations
        
        # GRU for region refinement
        self.gru = nn.GRUCell(input_size=feature_dim + hidden_dim, hidden_size=hidden_dim)
        
        # Decision heads
        self.add_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        self.remove_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        self.completion_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, features: torch.Tensor, region_mask: torch.Tensor, 
                adaptive_iterations: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, N, feature_dim) point features
            region_mask: (B, N) binary mask indicating region membership
            adaptive_iterations: (B,) number of iterations per batch sample
        Returns:
            Dictionary with keys:
                - add_probs: (B, N) probability of adding each point
                - remove_probs: (B, N) probability of removing each point
                - completion: (B,) region completion confidence
        """
        B, N, D = features.shape
        device = features.device
        
        if adaptive_iterations is None:
            adaptive_iterations = torch.full((B,), self.num_iterations, device=device, dtype=torch.long)
        
        # Initialize hidden state
        h = torch.zeros(B * N, self.hidden_dim, device=device)
        
        # Iteratively refine region
        add_probs_all = []
        remove_probs_all = []
        
        for iteration in range(self.num_iterations):
            # Aggregate neighborhood features
            region_mask_expanded = region_mask.unsqueeze(2).expand(-1, -1, D)
            region_features = features * region_mask_expanded  # Mask features
            region_summary = region_features.sum(dim=1, keepdim=True) / (region_mask.sum(dim=1, keepdim=True, dtype=torch.float32) + 1e-8)
            
            # Broadcast region summary
            region_summary_expanded = region_summary.expand(-1, N, -1)  # (B, N, D)
            
            # Concatenate with global region summary
            gru_input = torch.cat([features, region_summary_expanded], dim=-1)  # (B, N, 2D)
            gru_input_flat = gru_input.reshape(B * N, -1)
            
            # Update hidden state
            h = self.gru(gru_input_flat, h)
            
            # Predict add/remove probabilities
            add_prob = self.add_head(h).reshape(B, N, 1)
            remove_prob = self.remove_head(h).reshape(B, N, 1)
            
            add_probs_all.append(add_prob)
            remove_probs_all.append(remove_prob)
            
            # Update region mask (simplified: threshold-based)
            add_mask = (add_prob.squeeze(-1) > 0.5).float()
            remove_mask = (remove_prob.squeeze(-1) > 0.5).float()
            
            region_mask = region_mask * (1 - remove_mask) + add_mask * (1 - region_mask)
        
        # Final completion prediction
        completion = self.completion_head(h).reshape(B, N, 1)
        completion_per_region = completion.masked_fill(region_mask.unsqueeze(-1) == 0, 0).sum(dim=1) / (region_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        return {
            'add_probs': torch.stack(add_probs_all, dim=1).mean(dim=1).squeeze(-1),  # (B, N)
            'remove_probs': torch.stack(remove_probs_all, dim=1).mean(dim=1).squeeze(-1),  # (B, N)
            'completion': completion_per_region.squeeze(-1),  # (B,)
            'region_mask': region_mask,
        }


class AleatoricUncertaintyHead(nn.Module):
    """
    Predicts per-point aleatoric (data) uncertainty.
    Uses the probabilistic interpretation of uncertainty.
    """
    def __init__(self, feature_dim: int = 512):
        super(AleatoricUncertaintyHead, self).__init__()
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus(),  # Ensure positive values
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, feature_dim)
        Returns:
            uncertainty: (B, N) per-point aleatoric uncertainty
        """
        B, N, D = features.shape
        uncertainty = self.uncertainty_head(features.reshape(B * N, D))
        return uncertainty.reshape(B, N)


class InstanceSegmentationHead(nn.Module):
    """
    Instance segmentation head predicting instance labels.
    Uses embedding-based approach with clustering.
    """
    def __init__(self, feature_dim: int = 512, max_instances: int = 32):
        super(InstanceSegmentationHead, self).__init__()
        
        self.max_instances = max_instances
        
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),  # Low-dimensional embedding
        )
        
        self.center_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, N, feature_dim)
        Returns:
            embeddings: (B, N, 16) point embeddings
            centers: (B, K, 128) instance center features
        """
        B, N, D = features.shape
        
        # Get embeddings
        embeddings = self.embedding_head(features.reshape(B * N, D)).reshape(B, N, -1)
        
        # Get center features
        centers = self.center_head(features.reshape(B * N, D)).reshape(B, N, -1)
        
        return embeddings, centers


class EfficientSegNet(nn.Module):
    """
    Complete EfficientSegNet model combining:
    - Hierarchical encoder
    - Region growing module
    - Uncertainty quantification (epistemic + aleatoric)
    - Instance segmentation head
    """
    def __init__(self, 
                 in_channels: int = 3,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 downsample_ratio: float = 0.2,
                 num_mc_samples: int = 5,
                 max_instances: int = 32):
        super(EfficientSegNet, self).__init__()
        
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_mc_samples = num_mc_samples
        self.downsample_ratio = downsample_ratio
        
        # Feature extraction
        self.encoder = HierarchicalEncoder(in_channels, feature_dim, downsample_ratio)
        
        # Region growing
        self.region_growing = RegionGrowingModule(feature_dim, hidden_dim, num_iterations=6)
        
        # Uncertainty heads
        self.aleatoric_head = AleatoricUncertaintyHead(feature_dim)
        
        # Instance segmentation
        self.instance_head = InstanceSegmentationHead(feature_dim, max_instances)
        
        # Final instance assignment
        self.instance_classifier = nn.Sequential(
            nn.Linear(feature_dim + 16, 256),  # feature_dim + embedding_dim
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, max_instances),
        )
    
    def forward(self, points: torch.Tensor, 
                return_uncertainty: bool = True,
                mc_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional uncertainty quantification.
        
        Args:
            points: (B, N, 3) input point cloud
            return_uncertainty: whether to compute uncertainty estimates
            mc_samples: number of MC dropout samples (default: self.num_mc_samples)
        
        Returns:
            Dictionary with keys:
                - instance_labels: (B, N) predicted instance IDs
                - confidence: (B, N) per-point confidence scores
                - epistemic_uncertainty: (B, N) model uncertainty
                - aleatoric_uncertainty: (B, N) data uncertainty
                - embeddings: (B, N, 16) learned point embeddings
        """
        
        if mc_samples is None:
            mc_samples = self.num_mc_samples if self.training or return_uncertainty else 1
        
        B, N, _ = points.shape
        device = points.device
        
        if not return_uncertainty or mc_samples == 1:
            # Single forward pass (inference mode)
            features, coarse_confidence = self.encoder(points)
            
            # Region growing (simplified for single pass)
            region_mask = (coarse_confidence.squeeze(-1) > 0.7).float()
            growing_results = self.region_growing(features, region_mask)
            
            # Aleatoric uncertainty
            aleatoric = self.aleatoric_head(features)
            
            # Instance segmentation
            embeddings, centers = self.instance_head(features)
            
            # Combine features for classification
            combined = torch.cat([features, embeddings], dim=-1)
            instance_logits = self.instance_classifier(combined.reshape(B * N, -1))
            instance_labels = instance_logits.argmax(dim=1).reshape(B, N)
            
            # Confidence (epistemic = 0 for single pass)
            epistemic = torch.zeros_like(aleatoric)
            confidence = torch.exp(-(epistemic + aleatoric))
            
            return {
                'instance_labels': instance_labels,
                'confidence': confidence,
                'epistemic_uncertainty': epistemic,
                'aleatoric_uncertainty': aleatoric,
                'embeddings': embeddings,
                'coarse_confidence': coarse_confidence.squeeze(-1),
            }
        
        else:
            # Multiple forward passes for MC dropout uncertainty
            all_logits = []
            all_aleatoric = []
            all_embeddings = []
            all_coarse_confidence = []
            
            for _ in range(mc_samples):
                features, coarse_confidence = self.encoder(points)
                
                # Aleatoric uncertainty
                aleatoric = self.aleatoric_head(features)
                
                # Instance segmentation
                embeddings, centers = self.instance_head(features)
                
                # Combine features
                combined = torch.cat([features, embeddings], dim=-1)
                instance_logits = self.instance_classifier(combined.reshape(B * N, -1))
                instance_logits = instance_logits.reshape(B, N, -1)
                
                all_logits.append(instance_logits)
                all_aleatoric.append(aleatoric)
                all_embeddings.append(embeddings)
                all_coarse_confidence.append(coarse_confidence.squeeze(-1))
            
            # Stack MC samples
            all_logits = torch.stack(all_logits, dim=0)  # (MC, B, N, K)
            all_aleatoric = torch.stack(all_aleatoric, dim=0)  # (MC, B, N)
            all_embeddings = torch.stack(all_embeddings, dim=0)  # (MC, B, N, D)
            all_coarse_confidence = torch.stack(all_coarse_confidence, dim=0)  # (MC, B, N)
            
            # Compute epistemic uncertainty (variance of predictions)
            predictions = F.softmax(all_logits, dim=-1)  # (MC, B, N, K)
            mean_prediction = predictions.mean(dim=0)  # (B, N, K)
            epistemic = torch.var(predictions, dim=0).sum(dim=-1)  # (B, N)
            
            # Aleatoric uncertainty (average over MC samples)
            aleatoric = all_aleatoric.mean(dim=0)  # (B, N)
            
            # Instance labels (from mean prediction)
            instance_labels = mean_prediction.argmax(dim=-1)  # (B, N)
            
            # Embeddings (average over MC samples)
            embeddings = all_embeddings.mean(dim=0)  # (B, N, D)
            
            # Combined confidence
            confidence = torch.exp(-(epistemic + aleatoric))
            
            return {
                'instance_labels': instance_labels,
                'confidence': confidence,
                'epistemic_uncertainty': epistemic,
                'aleatoric_uncertainty': aleatoric,
                'embeddings': embeddings,
                'coarse_confidence': all_coarse_confidence.mean(dim=0),
                'mean_prediction': mean_prediction,
            }


class EfficientSegNetLoss(nn.Module):
    """
    Combined loss function for training EfficientSegNet.
    Includes segmentation loss, uncertainty loss, and Lovász-softmax regularization.
    """
    def __init__(self, num_instances: int = 32, lambda_seg: float = 1.0, 
                 lambda_aleatoric: float = 0.5, lambda_lovasz: float = 0.3):
        super(EfficientSegNetLoss, self).__init__()
        
        self.lambda_seg = lambda_seg
        self.lambda_aleatoric = lambda_aleatoric
        self.lambda_lovasz = lambda_lovasz
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor,
                uncertainty_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            predictions: model output dictionary
            targets: (B, N) ground truth instance labels
            uncertainty_targets: (B, N) optional uncertainty targets
        
        Returns:
            total_loss: combined loss value
        """
        
        instance_labels = predictions['instance_labels']
        confidence = predictions['confidence']
        aleatoric = predictions.get('aleatoric_uncertainty', None)
        
        B, N = instance_labels.shape
        
        # Segmentation loss
        seg_loss = self.ce_loss(
            instance_labels.reshape(B * N, -1),
            targets.reshape(B * N)
        )
        
        # Aleatoric uncertainty loss (if available)
        aleatoric_loss = torch.tensor(0.0, device=instance_labels.device)
        if aleatoric is not None and uncertainty_targets is not None:
            # Probabilistic loss: higher uncertainty for incorrect predictions
            is_correct = (instance_labels == targets).float()
            aleatoric_loss = torch.mean(
                (1 / (2 * aleatoric + 1e-8)) * (is_correct - 1).pow(2) + 
                0.5 * torch.log(aleatoric + 1e-8)
            )
        
        # Lovász-softmax loss (approximates IoU)
        lovasz_loss = self._lovasz_softmax_loss(instance_labels, targets)
        
        total_loss = (
            self.lambda_seg * seg_loss +
            self.lambda_aleatoric * aleatoric_loss +
            self.lambda_lovasz * lovasz_loss
        )
        
        return total_loss
    
    @staticmethod
    def _lovasz_softmax_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Simplified Lovász-softmax loss for instance segmentation.
        """
        B, N = predictions.shape
        
        # Convert to per-class predictions
        unique_labels = torch.unique(targets)
        losses = []
        
        for label in unique_labels:
            if label == -1:
                continue
            
            pred_mask = (predictions == label).float()
            target_mask = (targets == label).float()
            
            # Jaccard loss approximation
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum() - intersection
            
            if union > 0:
                jaccard_loss = 1.0 - (intersection / (union + 1e-8))
                losses.append(jaccard_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=predictions.device)


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = EfficientSegNet(
        in_channels=3,
        feature_dim=512,
        hidden_dim=256,
        num_mc_samples=5,
        max_instances=32
    )
    
    # Create dummy input
    B, N = 4, 1024  # Batch size 4, 1024 points
    points = torch.randn(B, N, 3)
    targets = torch.randint(0, 32, (B, N))
    
    # Forward pass
    print("Single forward pass (inference):")
    model.eval()
    with torch.no_grad():
        outputs = model(points, return_uncertainty=False)
        print(f"Instance labels shape: {outputs['instance_labels'].shape}")
        print(f"Confidence shape: {outputs['confidence'].shape}")
    
    print("\nMC dropout inference (uncertainty):")
    outputs = model(points, return_uncertainty=True, mc_samples=5)
    print(f"Epistemic uncertainty shape: {outputs['epistemic_uncertainty'].shape}")
    print(f"Aleatoric uncertainty shape: {outputs['aleatoric_uncertainty'].shape}")
    
    print("\nTraining mode:")
    model.train()
    outputs = model(points, return_uncertainty=True)
    
    # Loss computation
    loss_fn = EfficientSegNetLoss()
    loss = loss_fn(outputs, targets)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
