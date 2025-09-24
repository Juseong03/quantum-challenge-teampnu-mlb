#!/usr/bin/env python3
"""
PK/PD Model Prediction Script

This script allows you to make predictions using trained models from the TeamPNU2 system.
It handles model loading, data preprocessing, and prediction for both PK and PD tasks.

Usage:
    python predict.py --model_path results/runs/seed_exp_fe_mixup_ct_mc/dual_stage/resmlp-resmlp/s1 \
                      --data_path data/new_data.csv \
                      --output_path predictions.csv

    python predict.py --model_path results/runs/seed_exp_fe_mixup_ct_mc/dual_stage/resmlp-resmlp/s1 \
                      --data_path data/new_data.csv \
                      --uncertainty \
                      --output_path predictions_with_uncertainty.csv
"""

import sys
import argparse
import pickle
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from utils.helpers import get_device
from utils.factory import create_model
from data.loaders import load_estdata, use_feature_engineering
# from data.splits import prepare_for_split  # Not needed for prediction


class PKPDPredictor:
    """
    PK/PD Model Predictor
    
    Handles loading trained models and making predictions on new data.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to the model directory containing model.pth, scalers.pkl, config.json
            device: Device to use for inference ("auto", "cpu", "cuda:0", etc.)
        """
        self.model_path = Path(model_path)
        self.device = get_device() if device == "auto" else torch.device(device)
        
        # Load configuration
        self.config = self._load_config()
        
        # Load feature information first (needed for model creation)
        self.pk_features, self.pd_features = self._extract_features()
        
        # Load scalers
        self.pk_scaler, self.pd_scaler, self.pk_target_scaler, self.pd_target_scaler = self._load_scalers()
        
        # Load model
        self.model = self._load_model()
        
        print(f"✅ Predictor initialized successfully")
        print(f"   Model: {self.config.mode} mode with {self.config.encoder} encoder")
        print(f"   Device: {self.device}")
        print(f"   PK features: {len(self.pk_features)}")
        print(f"   PD features: {len(self.pd_features)}")
    
    def _load_config(self) -> Config:
        """Load model configuration"""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create Config object from dictionary
        config = Config()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _load_scalers(self) -> Tuple[Any, Any, Any, Any]:
        """Load PK and PD scalers (both feature and target scalers)"""
        scaler_path = self.model_path / "scalers.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            pk_scaler = pickle.load(f)
            pd_scaler = pickle.load(f)
            
            # Try to load target scalers (new format)
            try:
                pk_target_scaler = pickle.load(f)
                pd_target_scaler = pickle.load(f)
            except EOFError:
                # Old format without target scalers
                pk_target_scaler = None
                pd_target_scaler = None
                print("⚠️  Target scalers not found - using scaled predictions")
        
        return pk_scaler, pd_scaler, pk_target_scaler, pd_target_scaler
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model"""
        model_path = self.model_path / "model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model architecture
        model = create_model(self.config, {}, self.pk_features, self.pd_features)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _extract_features(self) -> Tuple[List[str], List[str]]:
        """Extract feature names from configuration or results"""
        # Try to get feature counts from results.json first
        results_path = self.model_path / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
                if 'model_info' in results:
                    model_info = results['model_info']
                    pk_feature_count = model_info.get('pk_features', 0)
                    pd_feature_count = model_info.get('pd_features', 0)
                    if pk_feature_count > 0 and pd_feature_count > 0:
                        # Generate feature names based on counts
                        pk_features = [f"feature_{i}" for i in range(pk_feature_count)]
                        pd_features = [f"feature_{i}" for i in range(pd_feature_count)]
                        return pk_features, pd_features
        
        # Try to get feature counts from model weights
        model_path = self.model_path / "model.pth"
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                model_state = checkpoint['model_state_dict']
                
                # Extract feature counts from model weights
                pk_input_dim = model_state['pk_model.encoder.net.0.weight'].shape[1]
                pd_input_dim = model_state['pd_model.encoder.net.0.weight'].shape[1]
                
                print(f"📊 Detected model feature counts: PK={pk_input_dim}, PD={pd_input_dim}")
                
                pk_features = [f"feature_{i}" for i in range(pk_input_dim)]
                pd_features = [f"feature_{i}" for i in range(pd_input_dim)]
                return pk_features, pd_features
            except Exception as e:
                print(f"⚠️  Could not extract features from model: {e}")
        
        # Fallback: estimate features based on configuration
        if self.config.use_feature_engineering:
            # Estimate feature counts for feature engineering (updated for optimized version)
            pk_features = [f"feature_{i}" for i in range(18)]  # Updated PK features
            pd_features = [f"feature_{i}" for i in range(18)]  # Updated PD features
        else:
            # Basic features without feature engineering (4 features)
            pk_features = [f"feature_{i}" for i in range(4)]
            pd_features = [f"feature_{i}" for i in range(4)]
        
        return pk_features, pd_features
    
    def preprocess_data(self, df: pd.DataFrame, use_fe: bool = False) -> Dict[str, torch.Tensor]:
        """
        Preprocess new data for prediction
        
        Args:
            df: DataFrame with new data
            
        Returns:
            Dictionary with preprocessed PK and PD data
        """
        print("🔄 Preprocessing data...")
        
        # Apply feature engineering if requested
        if use_fe:
            print("   Applying feature engineering...")
            # For feature engineering, we need to use the original data loading approach
            # This is a simplified version - in practice, you'd need to handle dose data properly
            df_final = df.copy()
            # Use the same features as training
            pk_features = self.pk_features
            pd_features = self.pd_features
            print(f"   Using training features - PK: {len(pk_features)}, PD: {len(pd_features)}")
        else:
            # Use the same feature names as training
            df_final = df.copy()
            
            # Select only the features that were used during training
            # We need to match the exact feature count and order
            available_features = [col for col in df_final.columns 
                                if col not in ['ID', 'TIME', 'DV', 'DVID']]
            
            # Use the first N features where N matches the training feature count
            pk_features = available_features[:len(self.pk_features)]
            pd_features = available_features[:len(self.pd_features)]
        
        print(f"   Using {len(pk_features)} PK features: {pk_features}")
        print(f"   Using {len(pd_features)} PD features: {pd_features}")
        
        # Split PK and PD data
        pk_df = df_final[df_final["DVID"] == 1].copy()
        pd_df = df_final[df_final["DVID"] == 2].copy()
        
        print(f"   PK data: {pk_df.shape}, PD data: {pd_df.shape}")
        
        # Preprocess PK data
        pk_data = {}
        if not pk_df.empty:
            X_pk = pk_df[pk_features].values.astype(np.float32)
            X_pk_scaled = self.pk_scaler.transform(X_pk)
            pk_data = {
                'x': torch.tensor(X_pk_scaled, device=self.device),
                'y': torch.zeros(X_pk_scaled.shape[0], 1, device=self.device)  # Dummy target
            }
        
        # Preprocess PD data
        pd_data = {}
        if not pd_df.empty:
            X_pd = pd_df[pd_features].values.astype(np.float32)
            X_pd_scaled = self.pd_scaler.transform(X_pd)
            pd_data = {
                'x': torch.tensor(X_pd_scaled, device=self.device),
                'y': torch.zeros(X_pd_scaled.shape[0], 1, device=self.device)  # Dummy target
            }
        
        return {'pk': pk_data, 'pd': pd_data}
    
    def predict(self, data: Dict[str, torch.Tensor], uncertainty: bool = False) -> Dict[str, Any]:
        """
        Make predictions on preprocessed data
        
        Args:
            data: Preprocessed data dictionary
            uncertainty: Whether to use uncertainty quantification
            
        Returns:
            Dictionary with predictions
        """
        print("🔮 Making predictions...")
        
        with torch.no_grad():
            if uncertainty and self.config.use_mc_dropout:
                print("   Using Monte Carlo Dropout for uncertainty quantification...")
                results = self.model.predict_with_uncertainty(data)
            else:
                results = self.model(data)
        
        # The model outputs are in scaled space
        # Use target scalers to convert back to original space
        predictions = {}
        
        for task in ['pk', 'pd']:
            if task in results and task in data:
                pred_scaled = results[task]['pred']
                
                # Convert predictions back to original space if target scaler is available
                if task == 'pk' and self.pk_target_scaler is not None:
                    pred_original = self.pk_target_scaler.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1))
                    predictions[task] = {
                        'predictions': pred_original.flatten(),
                        'scaled_predictions': pred_scaled.cpu().numpy().flatten()
                    }
                elif task == 'pd' and self.pd_target_scaler is not None:
                    pred_original = self.pd_target_scaler.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1))
                    predictions[task] = {
                        'predictions': pred_original.flatten(),
                        'scaled_predictions': pred_scaled.cpu().numpy().flatten()
                    }
                else:
                    # Fallback to scaled predictions if target scaler not available
                    predictions[task] = {
                        'predictions': pred_scaled.cpu().numpy().flatten(),
                        'scaled_predictions': pred_scaled.cpu().numpy().flatten()
                    }
                
                # Add uncertainty information if available
                if uncertainty and 'std' in results[task]:
                    std_scaled = results[task]['std']
                    if task == 'pk' and self.pk_target_scaler is not None:
                        std_original = self.pk_target_scaler.scale_ * std_scaled.cpu().numpy()
                    elif task == 'pd' and self.pd_target_scaler is not None:
                        std_original = self.pd_target_scaler.scale_ * std_scaled.cpu().numpy()
                    else:
                        std_original = std_scaled.cpu().numpy()
                    
                    predictions[task]['uncertainty'] = {
                        'std': std_original.flatten(),
                        'confidence_interval': 1.96 * std_original.flatten()
                    }
        
        return predictions
    
    def predict_from_csv(self, csv_path: str, uncertainty: bool = False, use_fe: bool = False) -> pd.DataFrame:
        """
        Make predictions from CSV file
        
        Args:
            csv_path: Path to CSV file with new data
            uncertainty: Whether to use uncertainty quantification
            
        Returns:
            DataFrame with predictions
        """
        print(f"📁 Loading data from {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"   Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(df, use_fe=use_fe)
        
        # Make predictions
        predictions = self.predict(preprocessed_data, uncertainty=uncertainty)
        
        # Create results DataFrame
        results = []
        
        for task in ['pk', 'pd']:
            if task in predictions:
                task_data = df[df["DVID"] == (1 if task == 'pk' else 2)].copy()
                task_data[f'{task.upper()}_PREDICTION'] = predictions[task]['predictions']
                
                if uncertainty and 'uncertainty' in predictions[task]:
                    task_data[f'{task.upper()}_STD'] = predictions[task]['uncertainty']['std']
                    task_data[f'{task.upper()}_CI_LOWER'] = (
                        predictions[task]['predictions'] - predictions[task]['uncertainty']['confidence_interval']
                    )
                    task_data[f'{task.upper()}_CI_UPPER'] = (
                        predictions[task]['predictions'] + predictions[task]['uncertainty']['confidence_interval']
                    )
                
                results.append(task_data)
        
        if results:
            final_results = pd.concat(results, ignore_index=True)
            final_results = final_results.sort_values(['ID', 'TIME', 'DVID'])
            return final_results
        else:
            print("⚠️  No predictions made - check data format")
            return pd.DataFrame()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PK/PD Model Prediction")
    parser.add_argument("--model_path", required=True, help="Path to trained model directory")
    parser.add_argument("--data_path", required=True, help="Path to CSV file with new data")
    parser.add_argument("--output_path", required=True, help="Path to save predictions")
    parser.add_argument("--uncertainty", action="store_true", help="Use uncertainty quantification")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--use_fe", action="store_true", help="Use feature engineering (must match training configuration)")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = PKPDPredictor(args.model_path, device=args.device)
        
        # Make predictions
        results_df = predictor.predict_from_csv(args.data_path, uncertainty=args.uncertainty, use_fe=args.use_fe)
        
        # Save results
        results_df.to_csv(args.output_path, index=False)
        print(f"✅ Predictions saved to {args.output_path}")
        
        # Print summary
        print("\n📊 Prediction Summary:")
        for task in ['pk', 'pd']:
            task_col = f'{task.upper()}_PREDICTION'
            if task_col in results_df.columns:
                task_data = results_df[results_df["DVID"] == (1 if task == 'pk' else 2)]
                if not task_data.empty:
                    print(f"   {task.upper()}: {len(task_data)} predictions")
                    print(f"      Mean: {task_data[task_col].mean():.4f}")
                    print(f"      Std:  {task_data[task_col].std():.4f}")
                    print(f"      Range: [{task_data[task_col].min():.4f}, {task_data[task_col].max():.4f}]")
        
        if args.uncertainty:
            print("\n🎯 Uncertainty Information:")
            for task in ['pk', 'pd']:
                std_col = f'{task.upper()}_STD'
                if std_col in results_df.columns:
                    task_data = results_df[results_df["DVID"] == (1 if task == 'pk' else 2)]
                    if not task_data.empty:
                        print(f"   {task.upper()} Average Uncertainty: {task_data[std_col].mean():.4f}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
