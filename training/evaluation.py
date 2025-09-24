"""
Evaluation methods for PK/PD training
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List


class ModelEvaluator:
    """Handle all evaluation logic"""
    
    def __init__(self, model, device, logger):
        self.model = model
        self.device = device
        self.logger = logger
    
    def evaluate_test_set(self, data_loaders: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Evaluate model on test set"""
        self.logger.info("Evaluating on test set...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        test_metrics = {}
        
        with torch.no_grad():
            if mode == "separate":
                # Evaluate PK and PD separately
                test_metrics.update(self._evaluate_pk_test(data_loaders['test_pk']))
                test_metrics.update(self._evaluate_pd_test(data_loaders['test_pd']))
            else:
                # Evaluate combined model
                test_metrics.update(self._evaluate_combined_test(data_loaders))
        
        self.logger.info(f"Test Results - PK RMSE: {test_metrics.get('test_pk_rmse', 'N/A'):.6f}, "
                        f"PD RMSE: {test_metrics.get('test_pd_rmse', 'N/A'):.6f}, "
                        f"PK R²: {test_metrics.get('test_pk_r2', 'N/A'):.4f}, "
                        f"PD R²: {test_metrics.get('test_pd_r2', 'N/A'):.4f}")
        
        return test_metrics
    
    def _evaluate_pk_test(self, test_loader) -> Dict[str, Any]:
        """Evaluate PK model on test set"""
        pk_losses = []
        pk_predictions = []
        pk_targets = []
        
        for batch in test_loader:
            # Handle tuple batch format
            if isinstance(batch, (list, tuple)):
                pk_features = batch[0].to(self.device)
                pk_target = batch[1].to(self.device)
                batch_dict = {'x': pk_features, 'y': pk_target}
            else:
                batch_dict = self._to_device(batch)
                pk_target = batch_dict['y']
            
            # Forward pass
            pk_results = self.model({'pk': batch_dict})
            pk_pred = pk_results['pk']['pred']
            
            # Calculate loss (ensure same shape)
            pk_pred_flat = pk_pred.squeeze() if pk_pred.dim() > 1 else pk_pred
            pk_target_flat = pk_target.squeeze() if pk_target.dim() > 1 else pk_target
            pk_loss = F.mse_loss(pk_pred_flat, pk_target_flat)
            pk_losses.append(pk_loss.item())
            
            # Store predictions and targets
            pk_predictions.append(pk_pred.cpu())
            pk_targets.append(pk_target.cpu())
        
        # Concatenate all predictions and targets
        pk_predictions = torch.cat(pk_predictions, dim=0)
        pk_targets = torch.cat(pk_targets, dim=0)
        
        # Calculate metrics
        pk_pred_flat = pk_predictions.squeeze() if pk_predictions.dim() > 1 else pk_predictions
        pk_target_flat = pk_targets.squeeze() if pk_targets.dim() > 1 else pk_targets
        pk_mse = F.mse_loss(pk_pred_flat, pk_target_flat).item()
        pk_rmse = np.sqrt(pk_mse)
        pk_mae = F.l1_loss(pk_pred_flat, pk_target_flat).item()
        
        # Calculate R²
        pk_ss_res = torch.sum((pk_target_flat - pk_pred_flat) ** 2)
        pk_ss_tot = torch.sum((pk_target_flat - torch.mean(pk_target_flat)) ** 2)
        pk_r2 = 1 - (pk_ss_res / pk_ss_tot) if pk_ss_tot > 0 else 0
        
        return {
            'test_pk_mse': pk_mse,
            'test_pk_rmse': pk_rmse,
            'test_pk_mae': pk_mae,
            'test_pk_r2': pk_r2.item()
        }
    
    def _evaluate_pd_test(self, test_loader) -> Dict[str, Any]:
        """Evaluate PD model on test set"""
        pd_losses = []
        pd_predictions = []
        pd_targets = []
        
        for batch in test_loader:
            # Handle tuple batch format
            if isinstance(batch, (list, tuple)):
                pd_features = batch[0].to(self.device)
                pd_target = batch[1].to(self.device)
                batch_dict = {'x': pd_features, 'y': pd_target}
            else:
                batch_dict = self._to_device(batch)
                pd_target = batch_dict['y']
            
            # Forward pass
            pd_results = self.model({'pd': batch_dict})
            pd_pred = pd_results['pd']['pred']
            
            # Calculate loss (ensure same shape)
            pd_pred_flat = pd_pred.squeeze() if pd_pred.dim() > 1 else pd_pred
            pd_target_flat = pd_target.squeeze() if pd_target.dim() > 1 else pd_target
            pd_loss = F.mse_loss(pd_pred_flat, pd_target_flat)
            pd_losses.append(pd_loss.item())
            
            # Store predictions and targets
            pd_predictions.append(pd_pred.cpu())
            pd_targets.append(pd_target.cpu())
        
        # Concatenate all predictions and targets
        pd_predictions = torch.cat(pd_predictions, dim=0)
        pd_targets = torch.cat(pd_targets, dim=0)
        
        # Calculate metrics
        pd_pred_flat = pd_predictions.squeeze() if pd_predictions.dim() > 1 else pd_predictions
        pd_target_flat = pd_targets.squeeze() if pd_targets.dim() > 1 else pd_targets
        pd_mse = F.mse_loss(pd_pred_flat, pd_target_flat).item()
        pd_rmse = np.sqrt(pd_mse)
        pd_mae = F.l1_loss(pd_pred_flat, pd_target_flat).item()
        
        # Calculate R²
        pd_ss_res = torch.sum((pd_target_flat - pd_pred_flat) ** 2)
        pd_ss_tot = torch.sum((pd_target_flat - torch.mean(pd_target_flat)) ** 2)
        pd_r2 = 1 - (pd_ss_res / pd_ss_tot) if pd_ss_tot > 0 else 0
        
        return {
            'test_pd_mse': pd_mse,
            'test_pd_rmse': pd_rmse,
            'test_pd_mae': pd_mae,
            'test_pd_r2': pd_r2.item()
        }
    
    def _evaluate_combined_test(self, data_loaders: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate combined model on test set"""
        pk_losses = []
        pd_losses = []
        pk_predictions = []
        pd_predictions = []
        pk_targets = []
        pd_targets = []
        
        # Evaluate PK test set
        for batch in data_loaders['test_pk']:
            # Handle tuple batch format
            if isinstance(batch, (list, tuple)):
                pk_features = batch[0].to(self.device)
                pk_target = batch[1].to(self.device)
                batch_dict = {'x': pk_features, 'y': pk_target}
            else:
                batch_dict = self._to_device(batch)
                pk_target = batch_dict['y']
            
            # Forward pass
            pk_results = self.model({'pk': batch_dict})
            pk_pred = pk_results['pk']['pred']
            
            # Calculate loss (ensure same shape)
            pk_pred_flat = pk_pred.squeeze() if pk_pred.dim() > 1 else pk_pred
            pk_target_flat = pk_target.squeeze() if pk_target.dim() > 1 else pk_target
            pk_loss = F.mse_loss(pk_pred_flat, pk_target_flat)
            pk_losses.append(pk_loss.item())
            
            # Store predictions and targets
            pk_predictions.append(pk_pred.cpu())
            pk_targets.append(pk_target.cpu())
        
        # Evaluate PD test set
        for batch in data_loaders['test_pd']:
            # Handle tuple batch format
            if isinstance(batch, (list, tuple)):
                pd_features = batch[0].to(self.device)
                pd_target = batch[1].to(self.device)
                batch_dict = {'x': pd_features, 'y': pd_target}
            else:
                batch_dict = self._to_device(batch)
                pd_target = batch_dict['y']
            
            # Forward pass
            pd_results = self.model({'pd': batch_dict})
            pd_pred = pd_results['pd']['pred']
            
            # Calculate loss (ensure same shape)
            pd_pred_flat = pd_pred.squeeze() if pd_pred.dim() > 1 else pd_pred
            pd_target_flat = pd_target.squeeze() if pd_target.dim() > 1 else pd_target
            pd_loss = F.mse_loss(pd_pred_flat, pd_target_flat)
            pd_losses.append(pd_loss.item())
            
            # Store predictions and targets
            pd_predictions.append(pd_pred.cpu())
            pd_targets.append(pd_target.cpu())
        
        # Calculate PK metrics
        pk_predictions = torch.cat(pk_predictions, dim=0)
        pk_targets = torch.cat(pk_targets, dim=0)
        pk_pred_flat = pk_predictions.squeeze() if pk_predictions.dim() > 1 else pk_predictions
        pk_target_flat = pk_targets.squeeze() if pk_targets.dim() > 1 else pk_targets
        pk_mse = F.mse_loss(pk_pred_flat, pk_target_flat).item()
        pk_rmse = np.sqrt(pk_mse)
        pk_mae = F.l1_loss(pk_pred_flat, pk_target_flat).item()
        pk_ss_res = torch.sum((pk_target_flat - pk_pred_flat) ** 2)
        pk_ss_tot = torch.sum((pk_target_flat - torch.mean(pk_target_flat)) ** 2)
        pk_r2 = 1 - (pk_ss_res / pk_ss_tot) if pk_ss_tot > 0 else 0
        
        # Calculate PD metrics
        pd_predictions = torch.cat(pd_predictions, dim=0)
        pd_targets = torch.cat(pd_targets, dim=0)
        pd_pred_flat = pd_predictions.squeeze() if pd_predictions.dim() > 1 else pd_predictions
        pd_target_flat = pd_targets.squeeze() if pd_targets.dim() > 1 else pd_targets
        pd_mse = F.mse_loss(pd_pred_flat, pd_target_flat).item()
        pd_rmse = np.sqrt(pd_mse)
        pd_mae = F.l1_loss(pd_pred_flat, pd_target_flat).item()
        pd_ss_res = torch.sum((pd_target_flat - pd_pred_flat) ** 2)
        pd_ss_tot = torch.sum((pd_target_flat - torch.mean(pd_target_flat)) ** 2)
        pd_r2 = 1 - (pd_ss_res / pd_ss_tot) if pd_ss_tot > 0 else 0
        
        return {
            'test_pk_mse': pk_mse,
            'test_pk_rmse': pk_rmse,
            'test_pk_mae': pk_mae,
            'test_pk_r2': pk_r2.item(),
            'test_pd_mse': pd_mse,
            'test_pd_rmse': pd_rmse,
            'test_pd_mae': pd_mae,
            'test_pd_r2': pd_r2.item()
        }
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if torch.is_tensor(v) else v for v in batch]
        else:
            return batch.to(self.device)
