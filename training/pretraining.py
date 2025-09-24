"""
Contrastive pretraining methods for PK/PD training
"""

import torch
from typing import Dict, Any, List
from pathlib import Path
from .pk_label_integration import PKLabelIntegrator, CrossModalContrastiveLearning


class ContrastivePretraining:
    """Handle contrastive pretraining logic"""
    
    def __init__(self, model, config, data_loaders, device, logger):
        self.model = model
        self.config = config
        self.data_loaders = data_loaders

        self.device = device
        self.logger = logger
        self.model_save_directory = None
        
        # PK label integration for better PD pretraining
        self.pk_integrator = PKLabelIntegrator(logger)
        self.cross_modal_learning = CrossModalContrastiveLearning(temperature=config.temperature)
    
    def set_model_save_directory(self, directory: Path):
        """Set model save directory"""
        self.model_save_directory = directory
    
    def contrastive_pretraining(self, epochs: int = 50, patience: int = 20) -> Dict[str, Any]:
        """Contrastive Learning Pretraining Phase with PK-PD Integration"""
        self.logger.info(f"Starting Enhanced Contrastive Pretraining - Epochs: {epochs}")
        
        # Train PK predictor for PD enhancement
        self._train_pk_predictor()
        
        # Pretraining state
        pretraining_history = {'contrastive_loss': [], 'learning_rate': []}
        
        # Pretraining optimizer (separate from main training)
        pretraining_optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate * 2, 
            weight_decay=1e-4
        )
        pretraining_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            pretraining_optimizer, T_max=epochs
        )

        best_contrastive_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        for epoch in range(epochs):
            contrastive_loss = self._enhanced_pretraining_epoch(pretraining_optimizer)
            
            pretraining_history['contrastive_loss'].append(contrastive_loss)
            pretraining_history['learning_rate'].append(pretraining_optimizer.param_groups[0]['lr'])
            
            pretraining_scheduler.step()
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                self.logger.info(
                    f"Pretraining Epoch {epoch:3d} | "
                    f"Contrastive Loss: {contrastive_loss:.6f} | "
                    f"Best Loss: {best_contrastive_loss:.6f}"
                )
            
            if contrastive_loss < best_contrastive_loss:
                best_contrastive_loss = contrastive_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_pretrained_model()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch} with best loss: {best_contrastive_loss:.6f}")
                    break
        
        self.logger.info(f"Enhanced Contrastive Pretraining completed - Best Loss: {best_contrastive_loss:.6f}")
        
        return {
            'best_contrastive_loss': best_contrastive_loss,
            'pretraining_history': pretraining_history,
            'epochs_pretrained': best_epoch + 1
        }
    
    def _pretraining_epoch(self, optimizer) -> float:
        """Single pretraining epoch using contrastive learning with PK label integration"""
        self.model.train()
        total_contrastive_loss = 0.0
        num_batches = 0
        pretraining_loaders = self._get_pretraining_loaders()
        
        # Process PK batches
        for batch_pk in pretraining_loaders[0]:  # PK loader
            optimizer.zero_grad()
            
            # Move batch to device
            batch_pk = self._to_device(batch_pk)
            batch_pk_dict = self._convert_batch_to_dict(batch_pk, 'pk')
            
            # Contrastive learning for PK
            if batch_pk_dict is not None:
                pk_contrastive_loss = self._compute_contrastive_loss(batch_pk_dict, 'pk')
                
                pk_contrastive_loss.backward()
                optimizer.step()
                
                total_contrastive_loss += pk_contrastive_loss.item()
                num_batches += 1
        
        # Process PD batches with PK label integration
        for batch_pd in pretraining_loaders[1]:  # PD loader
            optimizer.zero_grad()
            
            # Move batch to device
            batch_pd = self._to_device(batch_pd)
            batch_pd_dict = self._convert_batch_to_dict(batch_pd, 'pd')
            
            # Contrastive learning for PD with PK label integration
            if batch_pd_dict is not None:
                pd_contrastive_loss = self._compute_contrastive_loss_with_pk_labels(batch_pd_dict, 'pd')
                
                pd_contrastive_loss.backward()
                optimizer.step()
                
                total_contrastive_loss += pd_contrastive_loss.item()
                num_batches += 1
        
        return total_contrastive_loss / num_batches if num_batches > 0 else 0.0
    
    def _compute_contrastive_loss(self, batch_dict: Dict[str, Any], task: str) -> torch.Tensor:
        """Compute contrastive loss for specific task"""
        if not hasattr(self, 'contrastive_learning') or self.contrastive_learning is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get input data
        # Get encoder for the task
        if task == 'pk':
            x = batch_dict['x']
            if hasattr(self.model, 'pk_model') and self.model.pk_model is not None:
                encoder = self.model.pk_model['encoder']
            elif hasattr(self.model, 'pk_encoder'):
                encoder = self.model.pk_encoder
            elif hasattr(self.model, 'shared_encoder'):
                encoder = self.model.shared_encoder
            else:
                encoder = self.model.encoder
        elif task == 'pd':
            if self.config.mode in ["separate", "joint", "dual_stage", "integrated"]:
                x = torch.cat([batch_dict['x'], batch_dict['pk_labels']], dim=1)
            else:
                x = batch_dict['x']
            if hasattr(self.model, 'pd_model') and self.model.pd_model is not None:
                encoder = self.model.pd_model['encoder']
            elif hasattr(self.model, 'pd_encoder'):
                encoder = self.model.pd_encoder
            elif hasattr(self.model, 'shared_encoder'):
                encoder = self.model.shared_encoder
            else:
                encoder = self.model.encoder
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Check if encoder exists and has the right input dimension
        if encoder is None:
            return torch.tensor(0.0, device=x.device, requires_grad=True)

      
        # Compute SimCLR contrastive loss
        return self.contrastive_learning.contrastive_loss(x, encoder)
    
    def _compute_contrastive_loss_with_pk_labels(self, batch_dict: Dict[str, Any], task: str) -> torch.Tensor:
        """Compute contrastive loss for PD with PK label integration"""
        if not hasattr(self, 'contrastive_learning') or self.contrastive_learning is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get input data
        x = batch_dict['x']
        
        # For PD task, try to get PK labels if available
        if task == 'pd' and 'pk_labels' in batch_dict:
            pk_labels = batch_dict['pk_labels']
            # Concatenate PK labels to PD input for enhanced representation
            x_enhanced = torch.cat([x, pk_labels.unsqueeze(-1) if pk_labels.dim() == 1 else pk_labels], dim=1)
        else:
            x_enhanced = x
        
        # Get encoder for the task
        if task == 'pk':
            if hasattr(self.model, 'pk_model') and self.model.pk_model is not None:
                encoder = self.model.pk_model['encoder']
            elif hasattr(self.model, 'pk_encoder'):
                encoder = self.model.pk_encoder
            elif hasattr(self.model, 'shared_encoder'):
                encoder = self.model.shared_encoder
            else:
                encoder = self.model.encoder
        elif task == 'pd':
            if hasattr(self.model, 'pd_model') and self.model.pd_model is not None:
                encoder = self.model.pd_model['encoder']
            elif hasattr(self.model, 'pd_encoder'):
                encoder = self.model.pd_encoder
            elif hasattr(self.model, 'shared_encoder'):
                encoder = self.model.shared_encoder
            else:
                encoder = self.model.encoder
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Check if encoder exists
        if encoder is None:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # For enhanced input, we need to handle dimension mismatch
        try:
            # Test if encoder can handle the enhanced input
            with torch.no_grad():
                _ = encoder(x_enhanced[:1])  # Test with single sample
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                # Fall back to original input if enhanced input doesn't work
                self.logger.debug(f"Enhanced input failed, using original input: {e}")
                x_enhanced = x
            else:
                raise e
        
        # Compute SimCLR contrastive loss with enhanced input
        return self.contrastive_learning.contrastive_loss(x_enhanced, encoder)
    
    def _train_pk_predictor(self):
        """Train PK predictor using PK data (train + val, exclude test)"""
        self.logger.info("Training PK predictor for enhanced PD pretraining...")
        
        # Get PK data for training predictor (train + val)
        pk_batches = list(self.data_loaders.get('train_pk', [])) + list(self.data_loaders.get('val_pk', []))
        if not pk_batches:
            self.logger.warning("No PK data available for predictor training")
            return
        
        # Collect PK data
        pk_features_list = []
        pk_targets_list = []
        
        for batch in pk_batches:
            batch = self._to_device(batch)
            if isinstance(batch, (list, tuple)):
                x, y = batch
                batch_dict = {'x': x, 'y': y}
            else:
                batch_dict = batch
            
            pk_features_list.append(batch_dict['x'])
            pk_targets_list.append(batch_dict['y'])
        
        # Concatenate all PK data
        pk_features = torch.cat(pk_features_list, dim=0)
        pk_targets = torch.cat(pk_targets_list, dim=0)
        
        # Train PK predictor
        pk_data = {'x': pk_features, 'y': pk_targets}
        pd_data = {'x': pk_features, 'y': pk_targets}  # Dummy for now
        
        self.pk_integrator.train_pk_predictor(pk_data, pd_data)
        self.logger.info("PK predictor training completed")


    def _enhanced_pretraining_epoch(self, optimizer) -> float:
        """Enhanced pretraining epoch with PK-PD integration (train + val)"""
        self.model.train()
        total_contrastive_loss = 0.0
        num_batches = 0
        
        # Get enhanced PD batches with PK labels (train + val)
        enhanced_pd_batches = self._get_enhanced_pd_batches()
        
        # Process PK batches (train + val)
        all_pk_batches = list(self.data_loaders.get('train_pk', [])) + list(self.data_loaders.get('val_pk', []))
        for batch_pk in all_pk_batches:
            optimizer.zero_grad()
            
            batch_pk = self._to_device(batch_pk)
            batch_pk_dict = self._convert_batch_to_dict(batch_pk, 'pk')
            
            if batch_pk_dict is not None:
                pk_contrastive_loss = self._compute_contrastive_loss(batch_pk_dict, 'pk')
                pk_contrastive_loss.backward()
                optimizer.step()
                
                total_contrastive_loss += pk_contrastive_loss.item()
                num_batches += 1
        
        # Process enhanced PD batches
        for enhanced_pd_batch in enhanced_pd_batches:
            optimizer.zero_grad()
            
            total_pd_loss = self._compute_contrastive_loss(enhanced_pd_batch, 'pd')
            total_pd_loss.backward()
            optimizer.step()
            
            total_contrastive_loss += total_pd_loss.item()
            num_batches += 1
        
        return total_contrastive_loss / num_batches if num_batches > 0 else 0.0


    def _get_enhanced_pd_batches(self) -> List[Dict[str, Any]]:
        """Get enhanced PD batches with PK labels (train + val)"""
        enhanced_batches = []
        all_pd_batches = list(self.data_loaders.get('train_pd', [])) + list(self.data_loaders.get('val_pd', []))
        
        for batch_pd in all_pd_batches:
            batch_pd = self._to_device(batch_pd)
            batch_pd_dict = self._convert_batch_to_dict(batch_pd, 'pd')
            
            # Enhance with PK labels
            enhanced_batch = self.pk_integrator.enhance_pd_batch_with_pk_labels(batch_pd_dict)
            enhanced_batches.append(enhanced_batch)
        
        return enhanced_batches

        
    def _compute_cross_modal_loss(self, enhanced_pd_batch: Dict[str, Any]) -> torch.Tensor:
        """Compute cross-modal contrastive loss"""
        if not hasattr(self, 'contrastive_learning') or self.contrastive_learning is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Get PD features
        pd_x = enhanced_pd_batch['x']
        pk_labels = enhanced_pd_batch.get('pk_labels', None)
        
        # Get encoders
        if hasattr(self.model, 'pk_model') and self.model.pk_model is not None:
            pk_encoder = self.model.pk_model['encoder']
        elif hasattr(self.model, 'pk_encoder'):
            pk_encoder = self.model.pk_encoder
        elif hasattr(self.model, 'shared_encoder'):
            pk_encoder = self.model.shared_encoder
        else:
            pk_encoder = self.model.encoder
        
        if hasattr(self.model, 'pd_model') and self.model.pd_model is not None:
            pd_encoder = self.model.pd_model['encoder']
        elif hasattr(self.model, 'pd_encoder'):
            pd_encoder = self.model.pd_encoder
        elif hasattr(self.model, 'shared_encoder'):
            pd_encoder = self.model.shared_encoder
        else:
            pd_encoder = self.model.encoder
        
        # Get PK and PD features
        try:
            if pk_labels is not None:
                # Use PK labels as PK features
                pk_features = pk_labels.unsqueeze(-1) if pk_labels.dim() == 1 else pk_labels
            else:
                # Fallback to PK encoder
                pk_features = pk_encoder(pd_x)
            
            pd_features = pd_encoder(pd_x)
            
            # Compute cross-modal contrastive loss
            cross_modal_loss = self.cross_modal_learning.compute_cross_modal_loss(
                pk_features, pd_features
            )
            
            return cross_modal_loss
            
        except Exception as e:
            self.logger.debug(f"Cross-modal learning failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _get_pretraining_loaders(self) -> List[Any]:
        """Get data loaders for pretraining"""
        return [self.data_loaders['train_pk'], self.data_loaders['train_pd']]
    
    def _convert_batch_to_dict(self, batch: Any, task: str) -> Dict[str, Any]:
        """Convert batch to dictionary format"""
        if isinstance(batch, (list, tuple)):
            x, y = batch
            return {'x': x, 'y': y}
        elif isinstance(batch, dict):
            return batch
        else:
            # Handle other batch formats
            return {'x': batch, 'y': None}
    
    def _save_pretrained_model(self):
        """Save pretrained model"""
        if self.model_save_directory is None:
            return
        
        pretrained_path = self.model_save_directory / "pretrained_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pretraining_phase': True,
            'config': self.config
        }, pretrained_path)
    
    def load_pretrained_model(self):
        """Load pretrained model"""
        if self.model_save_directory is None:
            return
        
        pretrained_path = self.model_save_directory / "pretrained_model.pth"
        if pretrained_path.exists():
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Pretrained model loaded successfully")
        else:
            self.logger.warning("No pretrained model found")
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if torch.is_tensor(v) else v for v in batch]
        else:
            return batch.to(self.device)
    
    def set_contrastive_learning(self, contrastive_learning):
        """Set contrastive learning instance"""
        self.contrastive_learning = contrastive_learning
