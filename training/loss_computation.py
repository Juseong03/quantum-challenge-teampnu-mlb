"""
Loss computation methods for PK/PD training
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class LossComputation:
    """Handle all loss computation logic"""
    
    def __init__(self, config, data_augmentation, contrastive_learning):
        self.config = config
        self.data_augmentation = data_augmentation
        self.contrastive_learning = contrastive_learning
    
    def compute_loss(self, model, batch_pk, batch_pd, mode: str, is_training: bool = True):
        if mode == "separate":
            return self._compute_separate_loss(model, batch_pk, batch_pd, is_training)
        elif mode in ["joint", "dual_stage", "integrated"]:
            return self._compute_dual_branch_loss(model, batch_pk, batch_pd, is_training)
        elif mode == "shared":
            return self._compute_shared_loss(model, batch_pk, batch_pd, is_training)
        elif mode == "two_stage_shared":
            return self._compute_two_stage_shared_loss(model, batch_pk, batch_pd, is_training)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    # -------------------------------
    # Separate loss
    # -------------------------------
    def _compute_separate_loss(self, model, batch_pk, batch_pd, is_training=True):
        losses, preds, targets = {'pk': torch.tensor(0.0), 'pd': torch.tensor(0.0), 'pd_clf': torch.tensor(0.0)}, {}, {}
        pk_preds, pd_preds, pd_pred_clfs = [], [], []
        pk_targets, pd_targets, pd_targets_clfs = [], [], []

        # PK loss
        if batch_pk is not None:
            batch_pk_dict = self._apply_augmentation(
                self._prepare_batch(batch_pk), is_training=is_training
            )
            pk_results = model({'pk': {'x': batch_pk_dict['x'], 'y': batch_pk_dict['y']}})
            pk_pred = pk_results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)
            pk_loss = F.mse_loss(pk_pred, pk_target)
            pk_preds.append(pk_pred)
            pk_targets.append(pk_target)

            if self.config.use_aug_supervised and 'x_aug' in batch_pk_dict:
                pk_results_aug = model({'pk': {'x': batch_pk_dict['x_aug'], 'y': batch_pk_dict['y_aug']}})
                pk_pred_aug = pk_results_aug['pk']['pred']
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = F.mse_loss(pk_pred_aug, pk_target_aug)
                pk_loss = pk_loss + getattr(self.config, 'aug_lambda', 0.5) * pk_loss_aug
            losses['pk'] = pk_loss

        # PD loss
        if batch_pd is not None:
            batch_pd_dict = self._apply_augmentation(
                self._prepare_batch(batch_pd), is_training=is_training
            )
            pd_results = model({'pd': {'x': batch_pd_dict['x'], 'y': batch_pd_dict['y']}})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)
            pd_loss = F.mse_loss(pd_pred, pd_target)
            pd_preds.append(pd_pred)
            pd_targets.append(pd_target)

            if self.config.use_aug_supervised and 'x_aug' in batch_pd_dict:
                pd_results_aug = model({'pd': {'x': batch_pd_dict['x_aug'], 'y': batch_pd_dict['y_aug']}})
                pd_pred_aug = pd_results_aug['pd']['pred']
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = F.mse_loss(pd_pred_aug, pd_target_aug)
                pd_loss = pd_loss + getattr(self.config, 'aug_lambda', 0.5) * pd_loss_aug

            losses['pd'] = pd_loss

            if getattr(self.config, 'use_clf', False):
                pd_loss_clf, pd_pred_clf, pd_target_clf = self._compute_clf_loss(model, batch_pd_dict)
                pd_pred_clfs.append(pd_pred_clf)
                pd_targets_clfs.append(pd_target_clf)
                losses['pd_clf'] = pd_loss_clf

        # Safe total loss aggregation
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for v in losses.values():
            total_loss = total_loss + v
        losses['total'] = total_loss

        preds['pk'] = pk_preds
        preds['pd'] = pd_preds
        preds['pd_clf'] = pd_pred_clfs
        targets['pk'] = pk_targets
        targets['pd'] = pd_targets
        targets['pd_clf'] = pd_targets_clfs
        return losses, preds, targets
    
    # -------------------------------
    # Dual branch, shared, two-stage
    # (구조 동일 → total_loss 부분만 동일하게 보완)
    # -------------------------------
    def _compute_dual_branch_loss(self, model, batch_pk, batch_pd, is_training=True):
        losses, preds, targets = {}, {}, {}
        pk_preds, pd_preds, pd_pred_clfs = [], [], []
        pk_targets, pd_targets, pd_targets_clfs = [], [], []
        batch_dict = {}
        if batch_pk is not None:
            batch_dict['pk'] = self._apply_augmentation(self._prepare_batch(batch_pk), is_training=is_training)
        if batch_pd is not None:
            batch_dict['pd'] = self._apply_augmentation(self._prepare_batch(batch_pd), is_training=is_training)

        # PK loss
        if 'pk' in batch_dict:
            results = model({'pk': {'x': batch_dict['pk']['x'], 'y': batch_dict['pk']['y']}})
            pk_pred = results['pk']['pred']
            pk_target = batch_dict['pk']['y'].squeeze(-1)
            pk_loss = F.mse_loss(pk_pred, pk_target)
            losses['pk'] = pk_loss
            pk_preds.append(pk_pred)
            pk_targets.append(pk_target)
            
            if self.config.use_aug_supervised:
                pk_results_aug = model({'pk': {'x': batch_dict['pk']['x_aug'], 'y': batch_dict['pk']['y_aug']}})
                pk_pred_aug = pk_results_aug['pk']['pred']
                pk_target_aug = batch_dict['pk']['y_aug'].squeeze(-1)
                pk_loss_aug = F.mse_loss(pk_pred_aug, pk_target_aug)
                pk_loss = pk_loss + getattr(self.config, 'aug_lambda', 0.5) * pk_loss_aug
                losses['pk'] = pk_loss

        # PD loss
        if 'pd' in batch_dict:
            pd_results = model({'pd': {'x': batch_dict['pd']['x'], 'y': batch_dict['pd']['y']}})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_dict['pd']['y'].squeeze(-1)
            pd_loss = F.mse_loss(pd_pred, pd_target)
            losses['pd'] = pd_loss
            pd_preds.append(pd_pred)
            pd_targets.append(pd_target)
            if self.config.use_aug_supervised:
                pd_results_aug = model({'pd': {'x': batch_dict['pd']['x_aug'], 'y': batch_dict['pd']['y_aug']}})
                pd_pred_aug = pd_results_aug['pd']['pred']
                pd_target_aug = batch_dict['pd']['y_aug'].squeeze(-1)
                pd_loss_aug = F.mse_loss(pd_pred_aug, pd_target_aug)
                pd_loss = pd_loss + getattr(self.config, 'aug_lambda', 0.5) * pd_loss_aug
                losses['pd'] = pd_loss

            if getattr(self.config, 'use_clf', False):
                pd_loss_clf, pd_pred_clf, pd_target_clf = self._compute_clf_loss(model, batch_dict['pd'])
                pd_pred_clfs.append(pd_pred_clf)
                pd_targets_clfs.append(pd_target_clf)
                losses['pd_clf'] = pd_loss_clf

        # Safe total loss
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for v in losses.values():
            total_loss = total_loss + v
        losses['total'] = total_loss

        preds['pk'] = pk_preds
        preds['pd'] = pd_preds
        preds['pd_clf'] = pd_pred_clfs
        targets['pk'] = pk_targets
        targets['pd'] = pd_targets
        targets['pd_clf'] = pd_targets_clfs
        return losses, preds, targets
    
    def _compute_shared_loss(
        self, model, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any], is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Calculate shared mode loss with data augmentation and contrastive learning"""
        losses, preds, targets = {}, {}, {}
        pk_preds, pd_preds, pd_pred_clfs = [], [], []
        pk_targets, pd_targets, pd_targets_clfs = [], [], []

        # PK loss
        if batch_pk is not None:
            batch_pk_dict = self._prepare_batch(batch_pk)
            batch_pk_dict = self._apply_augmentation(batch_pk_dict, is_training=is_training)

            pk_results = model({'pk': batch_pk_dict})
            pk_pred = pk_results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)
            pk_loss = F.mse_loss(pk_pred, pk_target)
            pk_preds.append(pk_pred)
            pk_targets.append(pk_target)
            
            if self.config.use_aug_supervised:
                pk_results_aug = model({'pk': {'x': batch_pk_dict['x_aug'], 'y': batch_pk_dict['y_aug']}})
                pk_pred_aug = pk_results_aug['pk']['pred']
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = F.mse_loss(pk_pred_aug, pk_target_aug)
                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pk_loss = pk_loss + aug_lambda * pk_loss_aug

            losses['pk'] = pk_loss

        # PD loss
        if batch_pd is not None:
            batch_pd_dict = self._prepare_batch(batch_pd)
            batch_pd_dict = self._apply_augmentation(batch_pd_dict, is_training=is_training)

            pd_results = model({'pd': batch_pd_dict})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)
            pd_loss = F.mse_loss(pd_pred, pd_target)
            pd_preds.append(pd_pred)
            pd_targets.append(pd_target)
            
            if self.config.use_aug_supervised:
                pd_results_aug = model({'pd': {'x': batch_pd_dict['x_aug'], 'y': batch_pd_dict['y_aug']}})
                pd_pred_aug = pd_results_aug['pd']['pred']
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = F.mse_loss(pd_pred_aug, pd_target_aug)

                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pd_loss = pd_loss + aug_lambda * pd_loss_aug

            if self.config.use_clf:  # only for PD
                pd_loss_clf, pd_pred_clf, pd_target_clf = self._compute_clf_loss(model, batch_pd_dict)
                pd_pred_clfs.append(pd_pred_clf)
                pd_targets_clfs.append(pd_target_clf)
                losses['pd_clf'] = pd_loss_clf

            losses['pd'] = pd_loss

        # Safe total loss accumulation
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for v in losses.values():
            total_loss = total_loss + v
        losses['total'] = total_loss

        preds['pk'] = pk_preds
        preds['pd'] = pd_preds
        preds['pd_clf'] = pd_pred_clfs
        targets['pk'] = pk_targets
        targets['pd'] = pd_targets
        targets['pd_clf'] = pd_targets_clfs
        return losses, preds, targets

    def _compute_two_stage_shared_loss(
        self, model, batch_pk: Dict[str, Any], batch_pd: Dict[str, Any], is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Calculate two-stage shared mode loss with data augmentation and contrastive learning"""
        losses, preds, targets = {'pk': torch.tensor(0.0), 'pd': torch.tensor(0.0), 'pd_clf': torch.tensor(0.0)}, {}, {}
        pk_preds, pd_preds, pd_pred_clfs = [], [], []
        pk_targets, pd_targets, pd_targets_clfs = [], [], []

        # Stage 1: PK prediction
        if batch_pk is not None:
            batch_pk_dict = self._prepare_batch(batch_pk)
            batch_pk_dict = self._apply_augmentation(batch_pk_dict, is_training=is_training)

            pk_results = model({'pk': batch_pk_dict})
            pk_pred = pk_results['pk']['pred']
            pk_target = batch_pk_dict['y'].squeeze(-1)
            pk_loss = F.mse_loss(pk_pred, pk_target)
            pk_preds.append(pk_pred)
            pk_targets.append(pk_target)
            
            if self.config.use_aug_supervised:
                pk_results_aug = model({'pk': {'x': batch_pk_dict['x_aug'], 'y': batch_pk_dict['y_aug']}})
                pk_pred_aug = pk_results_aug['pk']['pred']
                pk_target_aug = batch_pk_dict['y_aug'].squeeze(-1)
                pk_loss_aug = F.mse_loss(pk_pred_aug, pk_target_aug)
                pk_loss = pk_loss + getattr(self.config, 'aug_lambda', 0.5) * pk_loss_aug
            losses['pk'] = pk_loss
                

        # Stage 2: PD prediction (PK information included)
        if batch_pd is not None:
            batch_pd_dict = self._prepare_batch(batch_pd)
            batch_pd_dict = self._apply_augmentation(batch_pd_dict, is_training=is_training)

            pd_results = model({'pd': batch_pd_dict})
            pd_pred = pd_results['pd']['pred']
            pd_target = batch_pd_dict['y'].squeeze(-1)
            pd_loss = F.mse_loss(pd_pred, pd_target)
            pd_preds.append(pd_pred)
            pd_targets.append(pd_target)
            
            if self.config.use_aug_supervised:
                pd_results_aug = model({'pd': {'x': batch_pd_dict['x_aug'], 'y': batch_pd_dict['y_aug']}})
                pd_pred_aug = pd_results_aug['pd']['pred']
                pd_target_aug = batch_pd_dict['y_aug'].squeeze(-1)
                pd_loss_aug = F.mse_loss(pd_pred_aug, pd_target_aug)

                aug_lambda = getattr(self.config, 'aug_lambda', 0.5)
                pd_loss = pd_loss + aug_lambda * pd_loss_aug

            if self.config.use_clf:  # only for PD
                pd_loss_clf, pd_pred_clf, pd_target_clf = self._compute_clf_loss(model, batch_pd_dict)
                pd_pred_clfs.append(pd_pred_clf)
                pd_targets_clfs.append(pd_target_clf)
                losses['pd_clf'] = pd_loss_clf

            losses['pd'] = pd_loss

        # Safe total loss accumulation
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for v in losses.values():
            total_loss = total_loss + v
        losses['total'] = total_loss

        preds['pk'] = pk_preds
        preds['pd'] = pd_preds
        preds['pd_clf'] = pd_pred_clfs
        targets['pk'] = pk_targets
        targets['pd'] = pd_targets
        targets['pd_clf'] = pd_targets_clfs
        return losses, preds, targets

    # -------------------------------
    # Classification loss
    # -------------------------------
    def _compute_clf_loss(self, model, batch: Dict[str, Any]) -> torch.Tensor:
        batch_dict = self._prepare_batch(batch)

        results = model({'pd': {'x': batch_dict['x'], 'y': batch_dict['y']}})
        pd_pred = results['pd']['pred']   # [B, C]
        pd_target_clf = batch_dict['y_clf']

        if pd_target_clf is None:
            raise ValueError("y_clf is None but classification loss was requested")

        # -------------------------
        # Case 1: class indices (모드 1)
        # -------------------------
        if pd_target_clf.dim() == 1 or (pd_target_clf.dim() == 2 and pd_target_clf.size(-1) == 1):
            pd_target_clf = pd_target_clf.view(-1).float()
            loss = F.cross_entropy(pd_pred, pd_target_clf)

        # -------------------------
        # Case 2: one-hot or soft labels (모드 2) 
        # -------------------------
        elif pd_target_clf.dim() == 2 and pd_target_clf.size(-1) == pd_pred.size(-1):
            pd_target_clf = pd_target_clf.float()
            loss = F.cross_entropy(pd_pred, pd_target_clf)

        else:
            raise ValueError(
                f"Unexpected y_clf shape {pd_target_clf.shape}, "
                f"expected [B] or [B, C={pd_pred.size(-1)}]"
            )

        return loss, pd_pred, pd_target_clf


    # -------------------------------
    # Helpers
    # -------------------------------
    def _prepare_batch(self, batch):
        """Convert batch to dictionary format"""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, y, y_clf = batch
            elif len(batch) == 2:
                x, y = batch
                y_clf = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            return {'x': x, 'y': y, 'y_clf': y_clf}
        elif isinstance(batch, dict):
            return batch
        else:
            raise ValueError("Unsupported batch format")

    def _apply_augmentation(self, batch_dict, is_training=True):
        use_aug_supervised = getattr(self.config, 'use_aug_supervised', False)
        if not self.data_augmentation.aug_method or not use_aug_supervised or not is_training:
            return batch_dict

        x_orig, y_orig = batch_dict['x'], batch_dict['y']
        x_aug, y_aug = self.data_augmentation.apply_augmentation(x_orig, y_orig)

        if torch.isnan(x_aug).any() or torch.isinf(x_aug).any():
            x_aug = torch.nan_to_num(x_aug, nan=0.0, posinf=1e3, neginf=-1e3)
        if torch.isnan(y_aug).any() or torch.isinf(y_aug).any():
            y_aug = torch.nan_to_num(y_aug, nan=0.0, posinf=1e3, neginf=0.0)

        y_aug = torch.clamp(y_aug, min=0.0)
        x_aug = torch.clamp(x_aug, min=-1e3, max=1e3)

        updated_batch = batch_dict.copy()
        updated_batch['x_aug'] = x_aug
        updated_batch['y_aug'] = y_aug
        return updated_batch
