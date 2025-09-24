"""
Unified PK/PD Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from utils.factory import build_encoder, build_head
from .encoders import BaseEncoder
from .heads import BaseHead


class UnifiedPKPDModel(nn.Module):
    """
    Unified PK/PD Model - supports all training modes
    """
    
    def __init__(
        self,
        config,
        pk_features: List[str],
        pd_features: List[str],
        pk_input_dim: int,
        pd_input_dim: int
    ):
        super().__init__()
        self.config = config
        self.mode = config.mode
        self.pk_features = pk_features
        self.pd_features = pd_features
        
        # Build model for each mode
        if self.mode == "separate":
            self._build_separate_models(pk_input_dim, pd_input_dim)
        elif self.mode in ["joint", "dual_stage", "integrated"]:
            self._build_dual_branch_model(pk_input_dim, pd_input_dim)
        elif self.mode == "shared":
            self._build_shared_model(pk_input_dim, pd_input_dim)
        elif self.mode == "two_stage_shared":
            self._build_two_stage_shared_model(pk_input_dim, pd_input_dim)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _build_separate_models(self, pk_input_dim: int, pd_input_dim: int):
        """Separate mode: PK and PD are completely separate models"""
        # PK model - PK-specific encoder
        pk_encoder_type = self.config.encoder_pk or self.config.encoder
        pk_encoder = self._create_encoder(pk_encoder_type, pk_input_dim)
        pk_head = self._create_head(self.config.head_pk, pk_encoder.out_dim, "pk")
        self.pk_model = nn.ModuleDict({
            'encoder': pk_encoder,
            'head': pk_head
        })
        
        # PD model - PD-specific encoder with PK prediction input (+1 dimension)
        pd_encoder_type = self.config.encoder_pd or self.config.encoder
        pd_input_dim_with_pk = pd_input_dim + 1  # PK prediction adds 1 dimension
        pd_encoder = self._create_encoder(pd_encoder_type, pd_input_dim_with_pk)
        pd_head = self._create_head(self.config.head_pd, pd_encoder.out_dim, "pd")
        self.pd_model = nn.ModuleDict({
            'encoder': pd_encoder,
            'head': pd_head
        })
        
        # Save encoder information for logging
        self.pk_encoder_type = pk_encoder_type
        self.pd_encoder_type = pd_encoder_type
    
    def _build_dual_branch_model(self, pk_input_dim: int, pd_input_dim: int):
        """Joint/Dual-stage/Integrated mode: PK and PD are separated branches"""
        # PK branch - PK-specific encoder
        pk_encoder_type = self.config.encoder_pk or self.config.encoder
        self.pk_encoder = self._create_encoder(pk_encoder_type, pk_input_dim)
        self.pk_head = self._create_head(self.config.head_pk, self.pk_encoder.out_dim, "pk")
        
        # PD branch - PD-specific encoder
        pd_encoder_type = self.config.encoder_pd or self.config.encoder
        
        # Adjust input dimensions based on mode
        if self.mode in ["joint", "dual_stage"]:
            # PD receives PK prediction as additional feature (+1 dimension)
            pd_input_dim_with_pk = pd_input_dim + 1
        elif self.mode == "integrated":
            # Integrated mode: PD encoder receives PK encoder output
            # PD encoder input dimension = PK encoder output dimension
            pd_input_dim_with_pk = self.pk_encoder.out_dim
        else:
            pd_input_dim_with_pk = pd_input_dim
        
        self.pd_encoder = self._create_encoder(pd_encoder_type, pd_input_dim_with_pk)
        self.pd_head = self._create_head(self.config.head_pd, self.pd_encoder.out_dim, "pd")
        
        # Save encoder information for logging
        self.pk_encoder_type = pk_encoder_type
        self.pd_encoder_type = pd_encoder_type
        
        # Connection options (used in joint mode)
        if self.mode == "joint":
            self._setup_joint_connections()
    
    def _build_shared_model(self, pk_input_dim: int, pd_input_dim: int):
        """Shared mode: common encoder"""
        # Common encoder (use larger input dimension)
        shared_input_dim = max(pk_input_dim, pd_input_dim)
        self.shared_encoder = self._create_encoder(self.config.encoder, shared_input_dim)
        
        # PK/PD-specific heads
        self.pk_head = self._create_head(self.config.head_pk, self.shared_encoder.out_dim, "pk")
        self.pd_head = self._create_head(self.config.head_pd, self.shared_encoder.out_dim, "pd")
        
        # Input projection layers
        if pk_input_dim != shared_input_dim:
            self.pk_proj = nn.Linear(pk_input_dim, shared_input_dim)
        if pd_input_dim != shared_input_dim:
            self.pd_proj = nn.Linear(pd_input_dim, shared_input_dim)
    
    def _build_two_stage_shared_model(self, pk_input_dim: int, pd_input_dim: int):
        """Two-stage shared mode: One shared encoder used in 2 stages"""
        # Shared encoder (used in both stages)
        self.shared_encoder = self._create_encoder(self.config.encoder, pk_input_dim)
        
        # Stage 1: PK prediction
        self.stage1_head = self._create_head(self.config.head_pk, self.shared_encoder.out_dim, "pk")
        
        # Stage 2: PD prediction (PK information included)
        # PD input dimension = PD features + PK prediction
        stage2_input_dim = pd_input_dim + 1
        # Projection layer to map PD input to PK input dimension for shared encoder
        self.pd_to_pk_proj = nn.Linear(stage2_input_dim, pk_input_dim)
        self.stage2_head = self._create_head(self.config.head_pd, self.shared_encoder.out_dim, "pd")
    
    def _create_encoder(self, encoder_type: str, input_dim: int) -> BaseEncoder:
        """Create encoder"""
        return build_encoder(encoder_type, input_dim, self.config)
    
    def _create_head(self, head_type: str, input_dim: int, branch: str) -> BaseHead:
        """Create head"""
        return build_head(head_type, input_dim, self.config)
    
    def _setup_joint_connections(self): 
        """Joint mode: setup PK-PD connections"""
        # Projection layer to pass PK information to PD
        self.pk_to_pd_proj = nn.Linear(self.pk_encoder.out_dim, self.pd_encoder.out_dim)
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass - different logic for each mode"""
        if self.mode == "separate":
            return self._forward_separate(batch)
        elif self.mode in ["joint", "dual_stage", "integrated"]:
            return self._forward_dual_branch(batch)
        elif self.mode == "shared":
            return self._forward_shared(batch)
        elif self.mode == "two_stage_shared":
            return self._forward_two_stage_shared(batch)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _forward_separate(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Separate mode forward - PK first, then PD with PK prediction"""
        results = {}
        
        # Step 1: PK prediction
        if 'pk' in batch:
            x_pk = batch['pk']['x']
            encoder_output = self.pk_model['encoder'](x_pk)
            # Handle tuple output from ResMLPMoEEncoder
            if isinstance(encoder_output, tuple):
                z_pk, aux_loss = encoder_output
            else:
                z_pk = encoder_output
            pk_outs = self.pk_model['head'](z_pk, batch['pk'])
            results['pk'] = {
                'pred': pk_outs['pred'],
                'z': z_pk,
                'outs': pk_outs
            }
        
        # Step 2: PD prediction with PK information
        if 'pd' in batch:
            x_pd = batch['pd']['x']
            
            # Add PK prediction to PD input
            if 'pk' in results:
                pk_pred = results['pk']['pred']
                # Convert PK prediction to correct dimension
                if pk_pred.dim() == 1:
                    pk_pred = pk_pred.unsqueeze(-1)  # [B] -> [B, 1]
                
                # Handle batch size mismatch
                if pk_pred.size(0) != x_pd.size(0):
                    if pk_pred.size(0) < x_pd.size(0):
                        # Pad PK prediction with zeros
                        padding = torch.zeros(x_pd.size(0) - pk_pred.size(0), pk_pred.size(1), device=x_pd.device)
                        pk_pred = torch.cat([pk_pred, padding], dim=0)
                    else:
                        # Truncate PK prediction
                        pk_pred = pk_pred[:x_pd.size(0)]
                
                x_pd_with_pk = torch.cat([x_pd, pk_pred], dim=-1)
            else:
                # If PK prediction is not available, fill with 0
                batch_size = x_pd.size(0)
                zero_pk = torch.zeros(batch_size, 1, device=x_pd.device)
                x_pd_with_pk = torch.cat([x_pd, zero_pk], dim=-1)
            
            encoder_output = self.pd_model['encoder'](x_pd_with_pk)
            # Handle tuple output from ResMLPMoEEncoder
            if isinstance(encoder_output, tuple):
                z_pd, aux_loss = encoder_output
            else:
                z_pd = encoder_output
            pd_outs = self.pd_model['head'](z_pd, batch['pd'])
            results['pd'] = {
                'pred': pd_outs['pred'],
                'z': z_pd,
                'outs': pd_outs
            }
        
        return results
    
    def _forward_dual_branch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Dual branch mode forward - different logic for joint/dual_stage/integrated"""
        results = {}
        
        if self.mode == "joint":
            return self._forward_joint(batch)
        elif self.mode == "dual_stage":
            return self._forward_dual_stage(batch)
        elif self.mode == "integrated":
            return self._forward_integrated(batch)
        else:
            raise ValueError(f"Unknown dual branch mode: {self.mode}")
    
    def _forward_joint(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Joint mode: PK and PD trained together with PK info passed to PD"""
        results = {}
        
        # PK branch
        if 'pk' in batch:
            x_pk = batch['pk']['x']
            encoder_output = self.pk_encoder(x_pk)
            if isinstance(encoder_output, tuple):
                z_pk, aux_loss = encoder_output
            else:
                z_pk = encoder_output
            pk_outs = self.pk_head(z_pk, batch['pk'])
            results['pk'] = {
                'pred': pk_outs['pred'],
                'z': z_pk,
                'outs': pk_outs
            }
        
        # PD branch with PK information
        if 'pd' in batch:
            x_pd = batch['pd']['x']
            
            # Add PK prediction as additional feature to PD input
            if 'pk' in results:
                pk_pred = results['pk']['pred'].detach()  # Detach to prevent gradient flow
                # Ensure PK prediction has the same batch size as PD input
                if pk_pred.size(0) != x_pd.size(0):
                    # If batch sizes don't match, use zero padding for missing samples
                    if pk_pred.size(0) < x_pd.size(0):
                        # Pad PK prediction with zeros
                        if pk_pred.dim() == 1:
                            padding = torch.zeros(x_pd.size(0) - pk_pred.size(0), device=x_pd.device)
                        else:
                            padding = torch.zeros(x_pd.size(0) - pk_pred.size(0), pk_pred.size(1), device=x_pd.device)
                        pk_pred = torch.cat([pk_pred, padding], dim=0)
                    else:
                        # Truncate PK prediction
                        pk_pred = pk_pred[:x_pd.size(0)]
                # Concatenate PK prediction to PD input
                if pk_pred.dim() == 1:
                    x_pd_with_pk = torch.cat([x_pd, pk_pred.unsqueeze(-1)], dim=-1)
                else:
                    x_pd_with_pk = torch.cat([x_pd, pk_pred], dim=-1)
            else:
                # If no PK prediction available, use zero padding
                pk_pred = torch.zeros(x_pd.size(0), 1, device=x_pd.device)
                x_pd_with_pk = torch.cat([x_pd, pk_pred], dim=-1)
            
            encoder_output = self.pd_encoder(x_pd_with_pk)
            if isinstance(encoder_output, tuple):
                z_pd, aux_loss = encoder_output
            else:
                z_pd = encoder_output
            pd_outs = self.pd_head(z_pd, batch['pd'])
            results['pd'] = {
                'pred': pd_outs['pred'],
                'z': z_pd,
                'outs': pd_outs
            }
        
        return results
    
    def _forward_dual_stage(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Dual stage mode: PK first, then PD with PK information"""
        results = {}
        
        # Stage 1: PK prediction
        if 'pk' in batch:
            x_pk = batch['pk']['x']
            encoder_output = self.pk_encoder(x_pk)
            if isinstance(encoder_output, tuple):
                z_pk, aux_loss = encoder_output
            else:
                z_pk = encoder_output
            pk_outs = self.pk_head(z_pk, batch['pk'])
            results['pk'] = {
                'pred': pk_outs['pred'],
                'z': z_pk,
                'outs': pk_outs
            }
        
        # Stage 2: PD prediction using PK information
        if 'pd' in batch:
            x_pd = batch['pd']['x']
            
            # Use PK prediction as additional feature
            if 'pk' in results:
                pk_pred = results['pk']['pred']
                # Ensure PK prediction has the same batch size as PD input
                if pk_pred.size(0) != x_pd.size(0):
                    # If batch sizes don't match, use zero padding for missing samples
                    if pk_pred.size(0) < x_pd.size(0):
                        # Pad PK prediction with zeros
                        if pk_pred.dim() == 1:
                            padding = torch.zeros(x_pd.size(0) - pk_pred.size(0), device=x_pd.device)
                        else:
                            padding = torch.zeros(x_pd.size(0) - pk_pred.size(0), pk_pred.size(1), device=x_pd.device)
                        pk_pred = torch.cat([pk_pred, padding], dim=0)
                    else:
                        # Truncate PK prediction
                        pk_pred = pk_pred[:x_pd.size(0)]
                # Concatenate PK prediction to PD input
                if pk_pred.dim() == 1:
                    x_pd_with_pk = torch.cat([x_pd, pk_pred.unsqueeze(-1)], dim=-1)
                else:
                    x_pd_with_pk = torch.cat([x_pd, pk_pred], dim=-1)
            else:
                # If no PK prediction available, use zero padding
                pk_pred = torch.zeros(x_pd.size(0), 1, device=x_pd.device)
                x_pd_with_pk = torch.cat([x_pd, pk_pred], dim=-1)
            
            encoder_output = self.pd_encoder(x_pd_with_pk)
            if isinstance(encoder_output, tuple):
                z_pd, aux_loss = encoder_output
            else:
                z_pd = encoder_output
            pd_outs = self.pd_head(z_pd, batch['pd'])
            results['pd'] = {
                'pred': pd_outs['pred'],
                'z': z_pd,
                'outs': pd_outs
            }
        
        return results
    
    def _forward_integrated(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Integrated mode: PK encoder output (z_pk) is used as input to PD encoder"""
        results = {}
        
        # Step 1: PK processing
        if 'pk' in batch:
            x_pk = batch['pk']['x']
            encoder_output = self.pk_encoder(x_pk)
            if isinstance(encoder_output, tuple):
                z_pk, aux_loss = encoder_output
            else:
                z_pk = encoder_output
            pk_outs = self.pk_head(z_pk, batch['pk'])
            results['pk'] = {
                'pred': pk_outs['pred'],
                'z': z_pk,
                'outs': pk_outs
            }
        
        # Step 2: PD processing with PK encoder output (z_pk)
        if 'pd' in batch:
            x_pd = batch['pd']['x']
            
            # Use PK encoder output (z_pk) as input to PD encoder
            if 'pk' in results:
                z_pk = results['pk']['z']  # PK encoder output
                
                # Handle batch size mismatch
                if z_pk.size(0) != x_pd.size(0):
                    if z_pk.size(0) < x_pd.size(0):
                        # Pad z_pk with zeros
                        padding = torch.zeros(x_pd.size(0) - z_pk.size(0), z_pk.size(1), device=x_pd.device)
                        z_pk = torch.cat([z_pk, padding], dim=0)
                    else:
                        # Truncate z_pk
                        z_pk = z_pk[:x_pd.size(0)]
                
                # Use z_pk as input to PD encoder
                encoder_output = self.pd_encoder(z_pk)
            else:
                # If PK encoder output is not available, use zero input
                batch_size = x_pd.size(0)
                zero_input = torch.zeros(batch_size, self.pd_encoder.in_dim, device=x_pd.device)
                encoder_output = self.pd_encoder(zero_input)
            
            if isinstance(encoder_output, tuple):
                z_pd, aux_loss = encoder_output
            else:
                z_pd = encoder_output
            pd_outs = self.pd_head(z_pd, batch['pd'])
            results['pd'] = {
                'pred': pd_outs['pred'],
                'z': z_pd,
                'outs': pd_outs
            }
        
        return results
    
    def _forward_shared(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Shared mode forward"""
        results = {}
        
        if 'pk' in batch:
            x_pk = batch['pk']['x']
            if hasattr(self, 'pk_proj'):
                x_pk = self.pk_proj(x_pk)
            encoder_output = self.shared_encoder(x_pk)
            # Handle tuple output from ResMLPMoEEncoder
            if isinstance(encoder_output, tuple):
                z_shared, aux_loss = encoder_output
            else:
                z_shared = encoder_output
            pk_outs = self.pk_head(z_shared, batch['pk'])
            results['pk'] = {
                'pred': pk_outs['pred'],
                'z': z_shared,
                'outs': pk_outs
            }
        
        if 'pd' in batch:
            x_pd = batch['pd']['x']
            if hasattr(self, 'pd_proj'):
                x_pd = self.pd_proj(x_pd)
            encoder_output = self.shared_encoder(x_pd)
            # Handle tuple output from ResMLPMoEEncoder
            if isinstance(encoder_output, tuple):
                z_shared, aux_loss = encoder_output
            else:
                z_shared = encoder_output
            pd_outs = self.pd_head(z_shared, batch['pd'])
            results['pd'] = {
                'pred': pd_outs['pred'],
                'z': z_shared,
                'outs': pd_outs
            }
        
        return results
    
    def _forward_two_stage_shared(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Two-stage shared mode: One shared encoder used in 2 stages"""
        results = {}
        
        # Stage 1: PK prediction using shared encoder
        if 'pk' in batch:
            x_pk = batch['pk']['x']
            encoder_output = self.shared_encoder(x_pk)
            # Handle tuple output from ResMLPMoEEncoder
            if isinstance(encoder_output, tuple):
                z_shared, aux_loss = encoder_output
            else:
                z_shared = encoder_output
            pk_outs = self.stage1_head(z_shared, batch['pk'])
            results['pk'] = {
                'pred': pk_outs['pred'],
                'z': z_shared,
                'outs': pk_outs
            }
        
        # Stage 2: PD prediction using same shared encoder
        if 'pd' in batch:
            x_pd = batch['pd']['x']
            # Add PK prediction result to PD input
            if 'pk' in results:
                pk_pred = results['pk']['pred']
                # Convert PK prediction result to correct dimension
                if pk_pred.dim() == 1:
                    pk_pred = pk_pred.unsqueeze(-1)  # [B] -> [B, 1]
                
                # Handle batch size mismatch
                if pk_pred.size(0) != x_pd.size(0):
                    if pk_pred.size(0) < x_pd.size(0):
                        # Pad PK prediction with zeros
                        padding = torch.zeros(x_pd.size(0) - pk_pred.size(0), pk_pred.size(1), device=x_pd.device)
                        pk_pred = torch.cat([pk_pred, padding], dim=0)
                    else:
                        # Truncate PK prediction
                        pk_pred = pk_pred[:x_pd.size(0)]
                
                x_pd_with_pk = torch.cat([x_pd, pk_pred], dim=-1)
            else:
                # If PK prediction is not available, fill with 0
                batch_size = x_pd.size(0)
                zero_pk = torch.zeros(batch_size, 1, device=x_pd.device)
                x_pd_with_pk = torch.cat([x_pd, zero_pk], dim=-1)
            
            # Project PD input to PK input dimension for shared encoder
            x_pd_projected = self.pd_to_pk_proj(x_pd_with_pk)
            # Use same shared encoder for PD
            encoder_output = self.shared_encoder(x_pd_projected)
            # Handle tuple output from ResMLPMoEEncoder
            if isinstance(encoder_output, tuple):
                z_shared_pd, aux_loss = encoder_output
            else:
                z_shared_pd = encoder_output
            pd_outs = self.stage2_head(z_shared_pd, batch['pd'])
            results['pd'] = {
                'pred': pd_outs['pred'],
                'z': z_shared_pd,
                'outs': pd_outs
            }
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'mode': self.mode,
            'total_parameters': total_params,
            'pk_features': len(self.pk_features),
            'pd_features': len(self.pd_features)
        }
        
        # Add encoder information
        if hasattr(self, 'pk_encoder_type') and hasattr(self, 'pd_encoder_type'):
            info['pk_encoder'] = self.pk_encoder_type
            info['pd_encoder'] = self.pd_encoder_type
        else:
            info['encoder'] = self.config.encoder
        
        if self.mode == "separate":
            info['pk_parameters'] = sum(p.numel() for p in self.pk_model.parameters() if p.requires_grad)
            info['pd_parameters'] = sum(p.numel() for p in self.pd_model.parameters() if p.requires_grad)
        
        return info
    
    def predict_with_uncertainty(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Predict with uncertainty using Monte Carlo Dropout
        
        This method enables dropout during inference and performs multiple forward passes
        to estimate prediction uncertainty. The model should have dropout layers for this to work.
        """
        if not self.config.use_mc_dropout:
            raise ValueError("MCDropout is not enabled. Set use_mc_dropout=True in config.")
        
        # Enable dropout for uncertainty estimation
        self.train()  # This enables dropout layers
        
        results = {}
        n_samples = self.config.mc_samples
        
        for branch in ['pk', 'pd']:
            if branch in batch:
                # Collect multiple predictions
                predictions = []
                
                for i in range(n_samples):
                    # Forward pass with dropout enabled
                    branch_results = self.forward({branch: batch[branch]})
                    pred = branch_results[branch]['pred']
                    predictions.append(pred.detach())
                
                # Stack predictions: [n_samples, batch_size]
                predictions = torch.stack(predictions, dim=0)
                
                # Calculate statistics
                mean_pred = predictions.mean(dim=0)
                std_pred = predictions.std(dim=0)
                confidence_interval = 1.96 * std_pred  # 95% confidence interval
                
                results[branch] = {
                    'pred': mean_pred,
                    'std': std_pred,
                    'confidence_interval': confidence_interval,
                    'all_predictions': predictions
                }
        
        # Disable dropout after uncertainty estimation
        self.eval()
        
        return results
