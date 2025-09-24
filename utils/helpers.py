"""
Helper functions for PK/PD modeling
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Additional utility functions
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from itertools import cycle
from models.heads import MSEHead
from models.encoders import MLPEncoder, ResMLPEncoder, MoEEncoder, ResMLPMoEEncoder
from models.encoders import QResMLPMoEEncoder, QMoEEncoder, QResMLPEncoder, QMLPEncoder, CNNEncoder

def generate_run_name(config) -> str:
    """
    Generate a descriptive run name based on configuration
    
    Args:
        config: Configuration object with training parameters
        
    Returns:
        str: Generated run name in format: mode_encoder_s{seed}_{timestamp}_{features}
    """
    # Create human-readable timestamp (YYMMDD_HHMM)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    
    # Determine encoder name for run name
    if config.encoder_pk or config.encoder_pd:
        # Different encoders for PK and PD
        pk_encoder = config.encoder_pk or config.encoder
        pd_encoder = config.encoder_pd or config.encoder
        encoder_name = f"{pk_encoder}-{pd_encoder}"
    else:
        # Same encoder for both
        encoder_name = config.encoder
    
    # Create postfix based on configuration (only add if features are enabled)
    postfix_parts = []
    
    # Feature engineering
    if getattr(config, 'use_feature_engineering', False):
        postfix_parts.append("fe")
    
    # Mixup augmentation
    if getattr(config, 'use_mixup', False):
        postfix_parts.append("mixup")
    
    # Contrastive learning
    if getattr(config, 'use_contrast', False):
        postfix_parts.append("contrast")
    
    # Uncertainty quantification
    if getattr(config, 'use_uncertainty', False):
        postfix_parts.append("uncertainty")
    
    # Active learning
    if getattr(config, 'use_active_learning', False):
        postfix_parts.append("active")
    
    # Meta learning
    if getattr(config, 'use_meta_learning', False):
        postfix_parts.append("meta")
    
    # Only add postfix if there are enabled features
    postfix = "_".join(postfix_parts) if postfix_parts else ""
    
    # Create clean run name
    if postfix:
        run_name = f"{config.mode}_{encoder_name}_s{config.random_state}_{timestamp}_{postfix}"
    else:
        run_name = f"{config.mode}_{encoder_name}_s{config.random_state}_{timestamp}"
    
    return run_name




def scaling_and_prepare_loader(data, features, batch_size, target_col):
    """Prepare data loader"""

    
    # Extract features and target
    if isinstance(data, dict):
        # Handle train/val/test splits
        train_data = data['train']
        val_data = data['val'] 
        test_data = data['test']
    else:
        # Single dataset - split it
        train_data = data
        val_data = data
        test_data = data
    
    # Prepare training data
    X_train = train_data[features].values.astype(np.float32)
    y_train = train_data[target_col].values.astype(np.float32).reshape(-1, 1)
    
    # Fit scaler on training data (optional scaling)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Option to disable scaling for testing
    use_scaling = True  # Set to False to disable scaling
    
    if use_scaling:
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
    else:
        X_train_scaled = X_train
        y_train_scaled = y_train
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled), 
        torch.tensor(y_train_scaled)
    )
    
    # Prepare validation data
    X_val = val_data[features].values.astype(np.float32)
    y_val = val_data[target_col].values.astype(np.float32).reshape(-1, 1)
    
    if use_scaling:
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val)
    else:
        X_val_scaled = X_val
        y_val_scaled = y_val
    
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled),
        torch.tensor(y_val_scaled)
    )
    
    # Prepare test data
    X_test = test_data[features].values.astype(np.float32)
    y_test = test_data[target_col].values.astype(np.float32).reshape(-1, 1)
    
    if use_scaling:
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        X_test_scaled = X_test
        y_test_scaled = y_test
    
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled),
        torch.tensor(y_test_scaled)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
    )
    
    return scaler_X, scaler_y, train_loader, val_loader, test_loader

class ReIter:
    """Re-iterable wrapper for data loaders"""
    def __init__(self, func, *loaders):
        self.func = func
        self.loaders = loaders
    
    def __iter__(self):
        return self.func(*self.loaders)

def roundrobin_loaders(*loaders):
    """Round-robin through multiple loaders"""
    # Create iterators for each loader
    iterators = [iter(loader) for loader in loaders]
    # Cycle through the iterators
    for iterator in cycle(iterators):
        try:
            yield next(iterator)
        except StopIteration:
            # If one loader is exhausted, remove it from the cycle
            iterators = [it for it in iterators if it is not iterator]
            if not iterators:
                break

def rr_val(*loaders):
    """Round-robin validation"""
    return roundrobin_loaders(*loaders)

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_device(device_id=0):
    """Return available device with specified device ID"""
    if torch.cuda.is_available():
        if device_id >= torch.cuda.device_count():
            print(f"Warning: Device {device_id} not available. Using device 0 instead.")
            device_id = 0
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """Calculate number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_feature_dimensions(config, branch="pk", pk_features=None, pd_features=None):
    """Get feature dimensions based on configuration and branch"""
    if hasattr(config, 'use_feature_engineering') and config.use_feature_engineering:
        # Use actual feature counts if provided
        if pk_features is not None and pd_features is not None:
            if branch == "pk":
                return len(pk_features)
            elif branch == "pd":
                return len(pd_features)
            else:
                return len(pd_features)  # Default to PD dimensions
        else:
            # Fallback to estimated counts (will be updated dynamically)
            if branch == "pk":
                return 25  # Estimated: basic features + window features + per-kg features
            elif branch == "pd":
                return 26  # Estimated: PK features + PD_BASELINE
            else:
                return 26  # Default to PD dimensions
    else:
        return 7  # Without feature engineering: both PK and PD use 7 features

def get_pk_input_dim(config):
    """Get PK input dimension"""
    return get_feature_dimensions(config, "pk")


def get_pd_input_dim(config):
    """Get PD input dimension"""
    return get_feature_dimensions(config, "pd")