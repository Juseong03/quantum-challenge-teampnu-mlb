"""
Unified PK/PD Trainer - Refactored and Modular
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import numpy as np

from utils.logging import get_logger
from utils.helpers import get_device
from models.heads import _reg_metrics
from utils.contrastive_learning import create_pkpd_contrastive_learning
from utils.data_augmentation import create_data_augmentation
from .loss_computation import LossComputation
from .evaluation import ModelEvaluator
from .pretraining import ContrastivePretraining


class UnifiedPKPDTrainer:
    """
    Unified PK/PD Trainer - All training modes supported (Refactored)
    """
    
    def __init__(self, model, config, data_loaders, device=None):
        self.model = model
        self.config = config
        self.data_loaders = data_loaders
        self.device = device if device is not None else get_device()
        self.logger = get_logger(__name__)
        self.mode = config.mode
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup components
        self._setup_optimizer()
        self._setup_model_save_directory()
        self._setup_components()
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_pk_rmse = float('inf')
        self.best_pd_rmse = float('inf')
        self.patience_counter = 0
        self.epoch = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'pk_train_loss': [], 'pd_train_loss': [], 
            'pk_val_loss': [], 'pd_val_loss': [],
            # Metrics
            'pk_train_mse': [], 'pk_train_rmse': [], 'pk_train_mae': [], 'pk_train_r2': [],
            'pd_train_mse': [], 'pd_train_rmse': [], 'pd_train_mae': [], 'pd_train_r2': [],
            'pk_val_mse': [], 'pk_val_rmse': [], 'pk_val_mae': [], 'pk_val_r2': [],
            'pd_val_mse': [], 'pd_val_rmse': [], 'pd_val_mae': [], 'pd_val_r2': []
        }
        
        self.logger.info(f"Unified Trainer initialized - Mode: {self.mode}, Device: {self.device}")
        self.logger.info(f"Number of model parameters: {self._count_parameters()}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=self.config.patience//2, factor=0.5, verbose=True
        )
    
    def _setup_model_save_directory(self):
        """Setup model save directory"""
        if self.config.encoder_pk or self.config.encoder_pd:
            pk_encoder = self.config.encoder_pk or self.config.encoder
            pd_encoder = self.config.encoder_pd or self.config.encoder
            encoder_name = f"{pk_encoder}-{pd_encoder}"
        else:
            encoder_name = self.config.encoder
        
        self.model_save_directory = Path(self.config.output_dir) / "models" / self.config.mode / encoder_name / f"s{self.config.random_state}"
        self.model_save_directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_components(self):
        """Setup training components"""
        # Data augmentation
        self.data_augmentation = create_data_augmentation(self.config)
        if self.data_augmentation.aug_method:
            self.logger.info(f"Data augmentation enabled: {self.data_augmentation.aug_method}")
        
        # Contrastive learning
        if self.config.lambda_contrast > 0:
            self.contrastive_learning = create_pkpd_contrastive_learning(self.config)
            self.logger.info(f"PK/PD Contrastive Learning enabled - Augmentation: {self.config.augmentation_type}")
        else:
            self.contrastive_learning = None
        
        # Loss computation
        self.loss_computation = LossComputation(
            self.model, self.config, self.data_augmentation, self.contrastive_learning
        )
        
        # Evaluation
        self.evaluator = ModelEvaluator(self.model, self.device, self.logger)
        
        # Pretraining
        self.pretraining = ContrastivePretraining(
            self.model, self.config, self.data_loaders, self.device, self.logger
        )
        self.pretraining.set_model_save_directory(self.model_save_directory)
        if self.contrastive_learning:
            self.pretraining.set_contrastive_learning(self.contrastive_learning)
    
    def _count_parameters(self) -> int:
        """Calculate number of model parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self) -> Dict[str, Any]:
        """Main training loop with optional contrastive pretraining"""
        self.logger.info(f"Training start - Mode: {self.mode}, Epochs: {self.config.epochs}")
        
        # Contrastive pretraining phase
        pretraining_results = None
        if self.config.lambda_contrast > 0 and getattr(self.config, 'use_contrastive_pretraining', False):
            self.logger.info("Starting Contrastive Pretraining Phase...")
            pretraining_epochs = getattr(self.config, 'contrastive_pretraining_epochs', 50)
            pretraining_results = self.pretraining.contrastive_pretraining(epochs=pretraining_epochs)
            
            # Load pretrained model for supervised training
            self.pretraining.load_pretrained_model()
            self._pretraining_completed = True  # Mark pretraining as completed
            self.logger.info("Pretrained model loaded, starting supervised training...")
        
        # Main supervised training
        if self.mode == "separate":
            return self._train_separate_mode(pretraining_results)
        else:
            return self._train_standard_mode(pretraining_results)
    
    def _train_standard_mode(self, pretraining_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Standard training loop for joint, shared, etc. modes"""
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self._train_epoch()
            
            # Validation
            val_metrics = self._validate_epoch()
            
            # Record metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total_loss'])
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['total_loss']):
                self.logger.info(f"Early stopping - Epoch {epoch}")
                break
            
            # Save best model
            if self._should_save_model(val_metrics):
                self._save_best_model()
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed - Time: {training_time:.2f} seconds")
        
        return self._get_final_results()
    
    def _train_separate_mode(self, pretraining_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Separate mode training: Train PK completely first, then PD"""
        start_time = time.time()
        
        # Phase 1: Train PK model completely
        self.logger.info("=== PHASE 1: Training PK Model ===")
        pk_results = self._train_pk_model()
        
        # Phase 2: Train PD model completely
        self.logger.info("=== PHASE 2: Training PD Model ===")
        pd_results = self._train_pd_model()
        
        training_time = time.time() - start_time
        self.logger.info(f"Separate training completed - Time: {training_time:.2f} seconds")
        
        # Evaluate test set for separate mode
        test_metrics = self.evaluator.evaluate_test_set(self.data_loaders, self.mode)
        
        # Combine results
        final_results = {
            'pk': pk_results,
            'pd': pd_results,
            'mode': 'separate',
            'training_time': training_time,
            'test_metrics': test_metrics
        }
        
        return final_results
    
    def _train_pk_model(self) -> Dict[str, Any]:
        """Train PK model completely"""
        self.logger.info("Starting PK model training...")
        
        # Reset best metrics for PK
        self.best_pk_rmse = float('inf')
        pk_epochs_without_improvement = 0
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train PK only
            train_metrics = self._train_pk_epoch()
            
            # Validate PK only
            val_metrics = self._validate_pk_epoch()
            
            # Log PK metrics
            self.logger.info(f"PK Epoch {epoch:3d} | Train | Loss: {train_metrics['pk_loss']:.6f} | RMSE: {train_metrics['pk_rmse']:.6f} | R²: {train_metrics['pk_r2']:.4f}")
            self.logger.info(f"PK Epoch {epoch:3d} | Valid | Loss: {val_metrics['pk_loss']:.6f} | RMSE: {val_metrics['pk_rmse']:.6f} | R²: {val_metrics['pk_r2']:.4f}")
            
            # Check for PK best model
            if val_metrics['pk_rmse'] < self.best_pk_rmse:
                self.best_pk_rmse = val_metrics['pk_rmse']
                self._save_best_model()
                pk_epochs_without_improvement = 0
                self.logger.info(f"New PK best model - RMSE: {self.best_pk_rmse:.6f}")
            else:
                pk_epochs_without_improvement += 1
            
            # Early stopping for PK
            if pk_epochs_without_improvement >= self.config.patience:
                self.logger.info(f"PK early stopping - Epoch {epoch}")
                break
        
        self.logger.info(f"PK training completed - Best RMSE: {self.best_pk_rmse:.6f}")
        return {'best_rmse': self.best_pk_rmse, 'epochs_trained': epoch + 1}
    
    def _train_pd_model(self) -> Dict[str, Any]:
        """Train PD model completely"""
        self.logger.info("Starting PD model training...")
        
        # Reset best metrics for PD
        self.best_pd_rmse = float('inf')
        pd_epochs_without_improvement = 0
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train PD only
            train_metrics = self._train_pd_epoch()
            
            # Validate PD only
            val_metrics = self._validate_pd_epoch()
            
            # Log PD metrics
            self.logger.info(f"PD Epoch {epoch:3d} | Train | Loss: {train_metrics['pd_loss']:.6f} | RMSE: {train_metrics['pd_rmse']:.6f} | R²: {train_metrics['pd_r2']:.4f}")
            self.logger.info(f"PD Epoch {epoch:3d} | Valid | Loss: {val_metrics['pd_loss']:.6f} | RMSE: {val_metrics['pd_rmse']:.6f} | R²: {val_metrics['pd_r2']:.4f}")
            
            # Check for PD best model
            if val_metrics['pd_rmse'] < self.best_pd_rmse:
                self.best_pd_rmse = val_metrics['pd_rmse']
                self._save_best_model()
                pd_epochs_without_improvement = 0
                self.logger.info(f"New PD best model - RMSE: {self.best_pd_rmse:.6f}")
            else:
                pd_epochs_without_improvement += 1
            
            # Early stopping for PD
            if pd_epochs_without_improvement >= self.config.patience:
                self.logger.info(f"PD early stopping - Epoch {epoch}")
                break
        
        self.logger.info(f"PD training completed - Best RMSE: {self.best_pd_rmse:.6f}")
        return {'best_rmse': self.best_pd_rmse, 'epochs_trained': epoch + 1}
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        if self.mode == "separate":
            return self._train_epoch_separate()
        else:
            return self._train_epoch_standard()
    
    def _train_epoch_standard(self) -> Dict[str, float]:
        """Train one epoch for standard modes (joint, shared, etc.)"""
        self.model.train()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        metrics_sum = {
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0,
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0
        }
        
        # Mixed precision settings
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Select mode-specific data loaders
        train_loaders = self._get_train_loaders()
        pk_loader, pd_loader = train_loaders
        
        # Process PK and PD batches independently
        pk_batches = list(pk_loader)
        pd_batches = list(pd_loader)
        
        # Process all PK batches
        for batch_pk in pk_batches:
            self.optimizer.zero_grad()
            batch_pk = self._to_device(batch_pk)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self.loss_computation.compute_loss(batch_pk, None, self.mode)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self.loss_computation.compute_loss(batch_pk, None, self.mode)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            total_loss += loss_dict['total'].item()
            pk_loss += loss_dict.get('pk', 0.0)
            num_batches += 1
        
        # Process all PD batches
        for batch_pd in pd_batches:
            self.optimizer.zero_grad()
            batch_pd = self._to_device(batch_pd)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self.loss_computation.compute_loss(None, batch_pd, self.mode)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self.loss_computation.compute_loss(None, batch_pd, self.mode)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            total_loss += loss_dict['total'].item()
            pd_loss += loss_dict.get('pd', 0.0)
            num_batches += 1
        
        # Calculate average
        result = {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches
        }
        
        return result
    
    def _train_epoch_separate(self) -> Dict[str, float]:
        """Train one epoch for separate mode - PK first, then PD"""
        self.model.train()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        metrics_sum = {
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0,
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0
        }
        
        # Mixed precision settings
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Phase 1: Train PK model
        self.logger.info("=== Phase 1: Training PK Model ===")
        pk_batch_count = 0
        for batch_pk in self.data_loaders['train_pk']:
            self.optimizer.zero_grad()
            batch_pk = self._to_device(batch_pk)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self.loss_computation.compute_loss(batch_pk, None, self.mode)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self.loss_computation.compute_loss(batch_pk, None, self.mode)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            total_loss += loss_dict['total'].item()
            pk_loss += loss_dict.get('pk', torch.tensor(0.0)).item()
            num_batches += 1
            pk_batch_count += 1
        
        self.logger.info(f"PK training completed - Processed {pk_batch_count} batches")
        
        # Phase 2: Train PD model
        self.logger.info("=== Phase 2: Training PD Model ===")
        pd_batch_count = 0
        for batch_pd in self.data_loaders['train_pd']:
            self.optimizer.zero_grad()
            batch_pd = self._to_device(batch_pd)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self.loss_computation.compute_loss(None, batch_pd, self.mode)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self.loss_computation.compute_loss(None, batch_pd, self.mode)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            total_loss += loss_dict['total'].item()
            pd_loss += loss_dict.get('pd', torch.tensor(0.0)).item()
            num_batches += 1
            pd_batch_count += 1
        
        self.logger.info(f"PD training completed - Processed {pd_batch_count} batches")
        
        # Average metrics
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches,
        }
        
        return avg_metrics
    
    def _train_pk_epoch(self) -> Dict[str, float]:
        """Train one epoch for PK only"""
        self.model.train()
        pk_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        metrics_sum = {
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0
        }
        
        # Mixed precision settings
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        for batch_pk in self.data_loaders['train_pk']:
            self.optimizer.zero_grad()
            batch_pk = self._to_device(batch_pk)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self.loss_computation.compute_loss(batch_pk, None, self.mode)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self.loss_computation.compute_loss(batch_pk, None, self.mode)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            pk_loss += loss_dict.get('pk', torch.tensor(0.0)).item()
            num_batches += 1
        
        return {
            'pk_loss': pk_loss / num_batches,
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0,
        }
    
    def _train_pd_epoch(self) -> Dict[str, float]:
        """Train one epoch for PD only"""
        self.model.train()
        pd_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        metrics_sum = {
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0
        }
        
        # Mixed precision settings
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        for batch_pd in self.data_loaders['train_pd']:
            self.optimizer.zero_grad()
            batch_pd = self._to_device(batch_pd)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = self.loss_computation.compute_loss(None, batch_pd, self.mode)
                scaler.scale(loss_dict['total']).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss_dict = self.loss_computation.compute_loss(None, batch_pd, self.mode)
                loss_dict['total'].backward()
                self.optimizer.step()
            
            pd_loss += loss_dict.get('pd', torch.tensor(0.0)).item()
            num_batches += 1
        
        return {
            'pd_loss': pd_loss / num_batches,
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0,
        }
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate one epoch"""
        if self.mode == "separate":
            return self._validate_epoch_separate()
        else:
            return self._validate_epoch_standard()
    
    def _validate_epoch_standard(self) -> Dict[str, float]:
        """Validate one epoch for standard modes"""
        self.model.eval()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        val_loaders = self._get_val_loaders()
        
        with torch.no_grad():
            for batch_pk, batch_pd in zip(*val_loaders):
                batch_pk = self._to_device(batch_pk)
                batch_pd = self._to_device(batch_pd)
                
                loss_dict = self.loss_computation.compute_loss(batch_pk, batch_pd, self.mode)
                
                total_loss += loss_dict['total'].item()
                pk_loss += loss_dict.get('pk', 0.0)
                pd_loss += loss_dict.get('pd', 0.0)
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches
        }
    
    def _validate_epoch_separate(self) -> Dict[str, float]:
        """Validate one epoch for separate mode - PK and PD separately"""
        self.model.eval()
        total_loss = 0.0
        pk_loss = 0.0
        pd_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            # Validate PK model
            for batch_pk in self.data_loaders['val_pk']:
                batch_pk = self._to_device(batch_pk)
                loss_dict = self.loss_computation.compute_loss(batch_pk, None, self.mode)
                
                total_loss += loss_dict['total'].item()
                pk_loss += loss_dict.get('pk', torch.tensor(0.0)).item()
                num_batches += 1
            
            # Validate PD model
            for batch_pd in self.data_loaders['val_pd']:
                batch_pd = self._to_device(batch_pd)
                loss_dict = self.loss_computation.compute_loss(None, batch_pd, self.mode)
                
                total_loss += loss_dict['total'].item()
                pd_loss += loss_dict.get('pd', torch.tensor(0.0)).item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'pk_loss': pk_loss / num_batches,
            'pd_loss': pd_loss / num_batches,
        }
    
    def _validate_pk_epoch(self) -> Dict[str, float]:
        """Validate one epoch for PK only"""
        self.model.eval()
        pk_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_pk in self.data_loaders['val_pk']:
                batch_pk = self._to_device(batch_pk)
                loss_dict = self.loss_computation.compute_loss(batch_pk, None, self.mode)
                
                pk_loss += loss_dict.get('pk', torch.tensor(0.0)).item()
                num_batches += 1
        
        return {
            'pk_loss': pk_loss / num_batches,
            'pk_mse': 0.0, 'pk_rmse': 0.0, 'pk_mae': 0.0, 'pk_r2': 0.0,
        }
    
    def _validate_pd_epoch(self) -> Dict[str, float]:
        """Validate one epoch for PD only"""
        self.model.eval()
        pd_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_pd in self.data_loaders['val_pd']:
                batch_pd = self._to_device(batch_pd)
                loss_dict = self.loss_computation.compute_loss(None, batch_pd, self.mode)
                
                pd_loss += loss_dict.get('pd', torch.tensor(0.0)).item()
                num_batches += 1
        
        return {
            'pd_loss': pd_loss / num_batches,
            'pd_mse': 0.0, 'pd_rmse': 0.0, 'pd_mae': 0.0, 'pd_r2': 0.0,
        }
    
    def _get_train_loaders(self) -> List[Any]:
        """Return training data loaders"""
        return [self.data_loaders['train_pk'], self.data_loaders['train_pd']]
    
    def _get_val_loaders(self) -> List[Any]:
        """Return validation data loaders"""
        return [self.data_loaders['val_pk'], self.data_loaders['val_pd']]
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if torch.is_tensor(v) else v for v in batch]
        else:
            return batch.to(self.device)
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics"""
        # Loss history recording
        self.training_history['train_loss'].append(train_metrics['total_loss'])
        self.training_history['val_loss'].append(val_metrics['total_loss'])
        self.training_history['pk_train_loss'].append(train_metrics.get('pk_loss', 0.0))
        self.training_history['pd_train_loss'].append(train_metrics.get('pd_loss', 0.0))
        self.training_history['pk_val_loss'].append(val_metrics.get('pk_loss', 0.0))
        self.training_history['pd_val_loss'].append(val_metrics.get('pd_loss', 0.0))
        
        # Log output
        self.logger.info(
            f"Epoch {epoch:4d} | Train |"
            f"Loss: {train_metrics['total_loss']:.6f} | "
            f"PK RMSE: {train_metrics.get('pk_rmse', 0.0):.6f} | "
            f"PD RMSE: {train_metrics.get('pd_rmse', 0.0):.6f}"
        )
        self.logger.info(
            f"Epoch {epoch:4d} | Valid |"
            f"Loss: {val_metrics['total_loss']:.6f} | "
            f"PK RMSE: {val_metrics.get('pk_rmse', 0.0):.6f} | "
            f"PD RMSE: {val_metrics.get('pd_rmse', 0.0):.6f}"
        )
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check early stopping"""
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def _should_save_model(self, val_metrics: Dict[str, float]) -> bool:
        """Check if model should be saved"""
        should_save = False
        
        if self.mode == "separate":
            # Separate mode: PK and PD each independently select best model
            if val_metrics.get('pk_rmse', float('inf')) < self.best_pk_rmse:
                self.best_pk_rmse = val_metrics.get('pk_rmse', float('inf'))
                should_save = True
                self.logger.info(f"New PK best model - RMSE: {self.best_pk_rmse:.6f}")
            
            if val_metrics.get('pd_rmse', float('inf')) < self.best_pd_rmse:
                self.best_pd_rmse = val_metrics.get('pd_rmse', float('inf'))
                should_save = True
                self.logger.info(f"New PD best model - RMSE: {self.best_pd_rmse:.6f}")
        else:
            # Other modes: PD RMSE selects best model, but also track PK RMSE
            if val_metrics.get('pd_rmse', float('inf')) < self.best_pd_rmse:
                self.best_pd_rmse = val_metrics.get('pd_rmse', float('inf'))
                self.best_pk_rmse = val_metrics.get('pk_rmse', float('inf'))
                should_save = True
                self.logger.info(f"New PD best model - RMSE: {self.best_pd_rmse:.6f}")
                self.logger.info(f"New PK best model - RMSE: {self.best_pk_rmse:.6f}")
        
        # Maintain existing total_loss criterion (for compatibility)
        if val_metrics['total_loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['total_loss']
            if not should_save:  # If not already saved
                should_save = True
        
        return should_save
    
    def _save_best_model(self):
        """Save best model"""
        model_path = self.model_save_directory / "best_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'val_loss': self.best_val_loss,
            'config': self.config
        }, model_path)
    
    def _get_final_results(self) -> Dict[str, Any]:
        """Return final results including test metrics"""
        # Evaluate test set
        test_metrics = self.evaluator.evaluate_test_set(self.data_loaders, self.mode)
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_pk_rmse': self.best_pk_rmse,
            'best_pd_rmse': self.best_pd_rmse,
            'final_epoch': self.epoch,
            'training_history': self.training_history,
            'model_info': self.model.get_model_info(),
            'test_metrics': test_metrics
        }
