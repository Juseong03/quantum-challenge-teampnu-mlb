"""
Loss computation methods for PK/PD training
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class LossComputation:
    """Handle all loss computation logic"""
    
    def __init__(self, model, config, data_augmentation, contrastive_learning):
        self.model = model
        self.config = config
        self.data_augmentation = data_augmentation
        self.contrastive_learning = contrastive_learning
    
    def compute_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any], mode: str, is_training: bool = True) -> Dict[str, torch.Tensor]:
        """Calculate loss - different logic for each mode"""
        if mode == "separate":
            return self._compute_separate_loss(batch_pk, batch_pd, is_training)
        elif mode in ["joint", "dual_stage", "integrated"]:
            return self._compute_dual_branch_loss(batch_pk, batch_pd, is_training)
        elif mode == "shared":
            return self._compute_shared_loss(batch_pk, batch_pd, is_training)
        elif mode == "two_stage_shared":
            return self._compute_two_stage_shared_loss(batch_pk, batch_pd, is_training)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute_separate_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any], is_training: bool = True) -> Dict[str, torch.Tensor]:
        """Calculate separate mode loss with data augmentation and contrastive learning"""
        losses = {}

        # PK loss
        if batch_pk is not None:
            batch_pk_dict = self._prepare_batch(batch_pk)
            batch_pk_dict = self._apply_augmentation(batch_pk_dict, is_training=is_training)

            # Calculate both original and augmented losses
            if 'x_orig' in batch_pk_dict:
                # Original loss
                pk_results_orig = self.model({'pk': {'x': batch_pk_dict['x_orig'], 'y': batch_pk_dict['y_orig']}})
                pk_pred_orig = pk_results_orig['pk']['pred']
                pk_target_orig = batch_pk_dict['y_orig'].squeeze(-1)
                pk_loss_orig = F.mse_loss(pk_pred_orig, pk_target_orig)

                # Augmented loss
                pk_results_aug = self.model({'pk': {'x': batch_pk_dict['x_aug'], 'y': batch_pk_dict['y_aug']}})
                pk_pred_aug = pk_results_aug['pk']['pred']
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = F.mse_loss(pk_pred_aug, pk_target_aug)

                # Combined loss: original + lambda * augmented
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pk_loss = pk_loss_orig + aug_lambda * pk_loss_aug
            else:
                # No augmentation case
                pk_results = self.model({'pk': batch_pk_dict})
                pk_pred = pk_results['pk']['pred']
                pk_target = batch_pk_dict['y'].squeeze(-1)
                pk_loss = F.mse_loss(pk_pred, pk_target)

            losses['pk'] = pk_loss
        
        # PD loss
        if batch_pd is not None:
            batch_pd_dict = self._prepare_batch(batch_pd)
            batch_pd_dict = self._apply_augmentation(batch_pd_dict, is_training=is_training)

            # Calculate both original and augmented losses
            if 'x_orig' in batch_pd_dict:
                # Original loss
                pd_results_orig = self.model({'pd': {'x': batch_pd_dict['x_orig'], 'y': batch_pd_dict['y_orig']}})
                pd_pred_orig = pd_results_orig['pd']['pred']
                pd_target_orig = batch_pd_dict['y_orig'].squeeze(-1)
                pd_loss_orig = F.mse_loss(pd_pred_orig, pd_target_orig)

                # Augmented loss
                pd_results_aug = self.model({'pd': {'x': batch_pd_dict['x_aug'], 'y': batch_pd_dict['y_aug']}})
                pd_pred_aug = pd_results_aug['pd']['pred']
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = F.mse_loss(pd_pred_aug, pd_target_aug)

                # Combined loss: original + lambda * augmented
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pd_loss = pd_loss_orig + aug_lambda * pd_loss_aug
            else:
                # No augmentation case
                pd_results = self.model({'pd': batch_pd_dict})
                pd_pred = pd_results['pd']['pred']
                pd_target = batch_pd_dict['y'].squeeze(-1)
                pd_loss = F.mse_loss(pd_pred, pd_target)

            losses['pd'] = pd_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_dual_branch_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any], is_training: bool = True) -> Dict[str, torch.Tensor]:
        """Calculate dual branch mode loss with data augmentation and contrastive learning"""
        losses = {}
        
        # Construct batch dictionary
        batch_dict = {}
        if batch_pk is not None:
            batch_pk_dict = self._prepare_batch(batch_pk)
            batch_pk_dict = self._apply_augmentation(batch_pk_dict, is_training=is_training)
            batch_dict['pk'] = batch_pk_dict
        
        if batch_pd is not None:
            batch_pd_dict = self._prepare_batch(batch_pd)
            batch_pd_dict = self._apply_augmentation(batch_pd_dict, is_training=is_training)
            batch_dict['pd'] = batch_pd_dict
        
        # Forward pass
        results = self.model(batch_dict)
        
        # PK loss
        if 'pk' in results:
            pk_pred = results['pk']['pred']
            pk_features = results['pk'].get('z', None)

            if 'x_orig' in batch_pk_dict:
                # Original loss
                pk_target_orig = batch_pk_dict['y_orig'].squeeze(-1)
                pk_loss_orig = F.mse_loss(pk_pred, pk_target_orig)

                # Augmented loss
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = F.mse_loss(pk_pred, pk_target_aug)

                # Combined loss
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pk_loss = pk_loss_orig + aug_lambda * pk_loss_aug
            else:
                # No augmentation case
                pk_target = batch_pk_dict['y'].squeeze(-1)
                pk_loss = F.mse_loss(pk_pred, pk_target)

            losses['pk'] = pk_loss

        # PD loss
        if 'pd' in results:
            pd_pred = results['pd']['pred']
            pd_features = results['pd'].get('z', None)

            if 'x_orig' in batch_pd_dict:
                # Original loss
                pd_target_orig = batch_pd_dict['y_orig'].squeeze(-1)
                pd_loss_orig = F.mse_loss(pd_pred, pd_target_orig)

                # Augmented loss
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = F.mse_loss(pd_pred, pd_target_aug)

                # Combined loss
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pd_loss = pd_loss_orig + aug_lambda * pd_loss_aug
            else:
                # No augmentation case
                pd_target = batch_pd_dict['y'].squeeze(-1)
                pd_loss = F.mse_loss(pd_pred, pd_target)

            losses['pd'] = pd_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_shared_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any], is_training: bool = True) -> Dict[str, torch.Tensor]:
        """Calculate shared mode loss with data augmentation and contrastive learning"""
        losses = {}
        
        # PK loss
        if batch_pk is not None:
            batch_pk_dict = self._prepare_batch(batch_pk)
            batch_pk_dict = self._apply_augmentation(batch_pk_dict, is_training=is_training)

            pk_results = self.model({'pk': batch_pk_dict})
            pk_pred = pk_results['pk']['pred']
            pk_features = pk_results['pk'].get('z', None)

            if 'x_orig' in batch_pk_dict:
                # Original loss
                pk_target_orig = batch_pk_dict['y_orig'].squeeze(-1)
                pk_loss_orig = F.mse_loss(pk_pred, pk_target_orig)

                # Augmented loss
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = F.mse_loss(pk_pred, pk_target_aug)

                # Combined loss
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pk_loss = pk_loss_orig + aug_lambda * pk_loss_aug
            else:
                # No augmentation case
                pk_target = batch_pk_dict['y'].squeeze(-1)
                pk_loss = F.mse_loss(pk_pred, pk_target)
            
            losses['pk'] = pk_loss

        # PD loss
        if batch_pd is not None:
            batch_pd_dict = self._prepare_batch(batch_pd)
            batch_pd_dict = self._apply_augmentation(batch_pd_dict, is_training=is_training)

            pd_results = self.model({'pd': batch_pd_dict})
            pd_pred = pd_results['pd']['pred']
            pd_features = pd_results['pd'].get('z', None)

            if 'x_orig' in batch_pd_dict:
                # Original loss
                pd_target_orig = batch_pd_dict['y_orig'].squeeze(-1)
                pd_loss_orig = F.mse_loss(pd_pred, pd_target_orig)

                # Augmented loss
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = F.mse_loss(pd_pred, pd_target_aug)

                # Combined loss
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pd_loss = pd_loss_orig + aug_lambda * pd_loss_aug
            else:
                # No augmentation case
                pd_target = batch_pd_dict['y'].squeeze(-1)
                pd_loss = F.mse_loss(pd_pred, pd_target)
            

            losses['pd'] = pd_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_two_stage_shared_loss(self, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any], is_training: bool = True) -> Dict[str, torch.Tensor]:
        """Calculate two-stage shared mode loss with data augmentation and contrastive learning"""
        losses = {}
        
        # Stage 1: PK prediction
        if batch_pk is not None:
            batch_pk_dict = self._prepare_batch(batch_pk)
            batch_pk_dict = self._apply_augmentation(batch_pk_dict, is_training=is_training)

            pk_results = self.model({'pk': batch_pk_dict})
            pk_pred = pk_results['pk']['pred']
            pk_features = pk_results['pk'].get('z', None)

            if 'x_orig' in batch_pk_dict:
                # Original loss
                pk_target_orig = batch_pk_dict['y_orig'].squeeze(-1)
                pk_loss_orig = F.mse_loss(pk_pred, pk_target_orig)

                # Augmented loss
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = F.mse_loss(pk_pred, pk_target_aug)

                # Combined loss
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pk_loss = pk_loss_orig + aug_lambda * pk_loss_aug
            else:
                # No augmentation case
                pk_target = batch_pk_dict['y'].squeeze(-1)
                pk_loss = F.mse_loss(pk_pred, pk_target)

            losses['pk'] = pk_loss

        # Stage 2: PD prediction (PK information included)
        if batch_pd is not None:
            batch_pd_dict = self._prepare_batch(batch_pd)
            batch_pd_dict = self._apply_augmentation(batch_pd_dict, is_training=is_training)

            pd_results = self.model({'pd': batch_pd_dict})
            pd_pred = pd_results['pd']['pred']
            pd_features = pd_results['pd'].get('z', None)

            if 'x_orig' in batch_pd_dict:
                # Original loss
                pd_target_orig = batch_pd_dict['y_orig'].squeeze(-1)
                pd_loss_orig = F.mse_loss(pd_pred, pd_target_orig)

                # Augmented loss
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = F.mse_loss(pd_pred, pd_target_aug)

                # Combined loss
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pd_loss = pd_loss_orig + aug_lambda * pd_loss_aug
            else:
                # No augmentation case
                pd_target = batch_pd_dict['y'].squeeze(-1)
                pd_loss = F.mse_loss(pd_pred, pd_target)
            

            losses['pd'] = pd_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _prepare_batch(self, batch: Any) -> Dict[str, Any]:
        """Convert batch to dictionary format"""
        if isinstance(batch, (list, tuple)):
            x, y = batch
            return {'x': x, 'y': y}
        elif isinstance(batch, dict):
            return batch
        else:
            return {'x': batch, 'y': None}
    
    def _apply_augmentation(self, batch_dict: Dict[str, Any], is_training: bool = True) -> Dict[str, Any]:
        """Apply data augmentation to batch (only during supervised training)"""
        use_aug_supervised = getattr(self.config, 'use_aug_supervised', False)

        if not self.data_augmentation.aug_method or not use_aug_supervised or not is_training:
            return batch_dict

        # Store original data
        x_orig = batch_dict['x']
        y_orig = batch_dict['y']

        # Apply augmentation
        x_aug, y_aug = self.data_augmentation.apply_augmentation(x_orig, y_orig)

        if torch.isnan(x_aug).any() or torch.isinf(x_aug).any():
            print("[Warning] NaN/Inf detected in x_aug, replacing with zeros")
            x_aug = torch.nan_to_num(x_aug, nan=0.0, posinf=1e3, neginf=-1e3)

        if torch.isnan(y_aug).any() or torch.isinf(y_aug).any():
            print("[Warning] NaN/Inf detected in y_aug, replacing with zeros")
            y_aug = torch.nan_to_num(y_aug, nan=0.0, posinf=1e3, neginf=0.0)

        # y (target)는 음수가 나오면 안 되므로 클리핑
        y_aug = torch.clamp(y_aug, min=0.0)

        # 입력도 극단적인 값 방지
        x_aug = torch.clamp(x_aug, min=-1e3, max=1e3)

        # Return updated batch
        updated_batch = batch_dict.copy()
        updated_batch['x_orig'] = x_orig
        updated_batch['y_orig'] = y_orig
        updated_batch['x_aug'] = x_aug
        updated_batch['y_aug'] = y_aug

        return updated_batch

    
    def _get_encoder(self, task: str):
        """Get encoder for specific task"""
        if task == 'pk':
            if hasattr(self.model, 'pk_model') and self.model.pk_model is not None:
                return self.model.pk_model['encoder']
            elif hasattr(self.model, 'pk_encoder'):
                return self.model.pk_encoder
            else:
                return self.model.encoder
        elif task == 'pd':
            if hasattr(self.model, 'pd_model') and self.model.pd_model is not None:
                return self.model.pd_model['encoder']
            elif hasattr(self.model, 'pd_encoder'):
                return self.model.pd_encoder
            else:
                return self.model.encoder
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _compute_contrastive_loss(self, features: torch.Tensor, encoder) -> torch.Tensor:
        """Compute contrastive loss"""
        if self.contrastive_learning is not None:
            return self.contrastive_learning.contrastive_loss(features, encoder)
        else:
            # Fallback to basic contrastive loss
            batch_size = features.size(0)
            if batch_size < 2:
                return torch.tensor(0.0, device=features.device, requires_grad=True)
            
            # Normalize features
            features = F.normalize(features, dim=1)
            
            # Calculate similarity matrix
            similarity_matrix = torch.matmul(features, features.T) / self.config.temperature
            
            # Diagonal mask (exclude self-similarity)
            mask = torch.eye(batch_size, device=features.device).bool()
            similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
            
            # Apply softmax
            logits = F.log_softmax(similarity_matrix, dim=1)
            
            # Ground truth labels (next sample)
            labels = torch.arange(batch_size, device=features.device)
            labels = (labels + 1) % batch_size
            
            # Loss calculation
            loss = F.nll_loss(logits, labels)
            return loss
