#!/usr/bin/env python3
"""
AutoVisualization Script
"""

import sys
import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import math

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.helpers import get_device
from utils.factory import create_model
from data.loaders import load_estdata, use_feature_engineering

class AutoVisualizer:    
    def __init__(self, run_dir: str, device: str = "auto"):
        """
        Args:
            run_dir: Experiment result directory (e.g. results/runs/experiment_name/shared/resmlp_moe-resmlp_moe/s1)
            device: Device setting
        """
        self.run_dir = Path(run_dir)
        # Extract device ID from torch.device object
        if isinstance(device, torch.device):
            if device.type == 'cuda':
                device_id = int(device.index) if device.index is not None else 0
            else:
                device_id = 0
        elif isinstance(device, str):
            if device == 'auto':
                device_id = 0
            elif device == 'cpu':
                device_id = 0
            elif device.startswith('cuda:'):
                device_id = int(device.split(':')[1])
            else:
                device_id = 0
        else:
            device_id = device
        self.device = get_device(device_id)
        
        # File paths
        self.model_path = self.run_dir / "model.pth"
        self.config_path = self.run_dir / "config.json"
        self.scalers_path = self.run_dir / "scalers.pkl"
        self.split_info_path = self.run_dir / "split_info.json"
        self.results_path = self.run_dir / "results.json"
        
        # Loaded data
        self.config = None
        self.model = None
        self.scalers = None
        self.split_info = None
        self.results = None
        
        print(f" AutoVisualizer initialized: {self.run_dir}")
        
    def load_experiment_data(self):
        """Loading experiment data"""
        print(" Loading experiment data...")
        
        # Config load
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        print(f"    Config loaded")
        
        # Scaler load
        with open(self.scalers_path, 'rb') as f:
            self.scalers = {
                'pk_scaler': pickle.load(f),
                'pd_scaler': pickle.load(f),
                'pk_target_scaler': pickle.load(f),
                'pd_target_scaler': pickle.load(f)
            }
        print(f"    Scaler loaded")
        
        # Split info load
        with open(self.split_info_path, 'r') as f:
            self.split_info = json.load(f)
        print(f"    Split info loaded")
        
        # Results load
        with open(self.results_path, 'r') as f:
            self.results = json.load(f)
        print(f"    Results loaded")
        
    def load_model(self):
        """Loading model"""
        print(" Loading model...")
        
        # Model state load
        model_state = torch.load(self.model_path, map_location=self.device)
        
        # Model is saved in state_dict format
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
            # Use the entire model object if it exists (for AutoVisualizer)
            if 'model' in model_state:
                self.model = model_state['model']
                print(f"    Use the entire model object (for AutoVisualizer compatibility)")
            else:
                # Only state_dict is available
                self.model = model_state['model_state_dict']
                print(f"    State dict model extracted")
                print(f"    State dict model cannot be used directly in AutoVisualizer.")
                return
        else:
            self.model = model_state
        
        print(f"    Model loaded")
        
    def prepare_test_data(self):
        """Preparing test data"""
        print(" Test data preparation...")
        
        # Original data load
        df_all, df_obs, df_dose = load_estdata(self.config['csv_path'])
        
        # Feature engineering applied
        if self.config.get('use_feature_engineering', False):
            df_final, pk_features, pd_features = use_feature_engineering(
                df_obs=df_obs, df_dose=df_dose,
                use_perkg=self.config.get('perkg', True),
                target="dv",
                allow_future_dose=self.config.get('allow_future_dose', True),
                time_windows=self.config.get('time_windows', [24, 48, 72, 96, 120, 144, 168])
            )
        else:
            df_final = df_obs.copy()
            feature_cols = [col for col in df_final.columns 
                          if col not in ['ID', 'TIME', 'DV', 'DVID']]
            pk_features = feature_cols
            pd_features = feature_cols
        
        # PK/PD data separation
        pk_df = df_final[df_final["DVID"] == 1].copy()
        pd_df = df_final[df_final["DVID"] == 2].copy()
        
        # Test data filtering (get test subjects from split_info)
        pk_test_subjects = self.split_info['pk_test_subjects']
        pd_test_subjects = self.split_info['pd_test_subjects']
        
        pk_test_data = pk_df[pk_df['ID'].isin(pk_test_subjects)].copy()
        pd_test_data = pd_df[pd_df['ID'].isin(pd_test_subjects)].copy()
        
        print(f"   PK Test data: {pk_test_data.shape}")
        print(f"   PD Test data: {pd_test_data.shape}")
        
        return pk_test_data, pd_test_data, pk_features, pd_features
    
    def make_predictions(self, test_data, features, target_scaler, data_type):
        """Making predictions"""
        print(f" {data_type.upper()} prediction...")
        
        # Feature data preparation
        X = test_data[features].values
        y_true = test_data['DV'].values
        
        # Scaling
        if data_type == 'pk':
            scaler = self.scalers['pk_scaler']
        else:
            scaler = self.scalers['pd_scaler']
        
        X_scaled = scaler.transform(X)
        
        # Tensor conversion
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_true).to(self.device)
        
        # Prediction
        with torch.no_grad():
            if hasattr(self.model, 'pk_model') and hasattr(self.model, 'pd_model'):
                # UnifiedPKPDModel in separate mode
                if data_type == 'pk':
                    # PK model: encoder -> head sequentially
                    encoded = self.model.pk_model['encoder'](X_tensor)
                    # MSEHead requires batch argument, so pass empty dictionary
                    head_output = self.model.pk_model['head'](encoded, {})
                    predictions = head_output['pred'].cpu().numpy().flatten()
                elif data_type == 'pd':
                    # PD model: first calculate PK prediction result and add to PD input
                    # 1. PK prediction
                    pk_encoded = self.model.pk_model['encoder'](X_tensor)
                    pk_head_output = self.model.pk_model['head'](pk_encoded, {})
                    pk_pred = pk_head_output['pred']
                    
                    # 2. PK prediction result to PD input (add to last dimension)
                    pd_input = torch.cat([X_tensor, pk_pred.unsqueeze(-1)], dim=-1)
                    
                    # 3. PD model: encoder -> head sequentially
                    encoded = self.model.pd_model['encoder'](pd_input)
                    # MSEHead requires batch argument, so pass empty dictionary
                    head_output = self.model.pd_model['head'](encoded, {})
                    predictions = head_output['pred'].cpu().numpy().flatten()
                else:
                    raise ValueError(f"Unknown data_type: {data_type}")
            elif isinstance(self.model, dict):
                # Model is a dictionary
                model = self.model[data_type]
                predictions = model(X_tensor).cpu().numpy().flatten()
            else:
                # Model is a single model - check if it's dual_stage mode
                if hasattr(self.model, 'mode') and self.model.mode == 'dual_stage':
                    # For dual_stage mode, create proper batch dictionary
                    batch_dict = {}
                    if data_type == 'pk':
                        batch_dict['pk'] = {'x': X_tensor, 'y': y_tensor}
                    elif data_type == 'pd':
                        batch_dict['pd'] = {'x': X_tensor, 'y': y_tensor}
                    
                    results = self.model(batch_dict)
                    if data_type == 'pk':
                        predictions = results['pk']['pred'].cpu().numpy().flatten()
                    elif data_type == 'pd':
                        predictions = results['pd']['pred'].cpu().numpy().flatten()
                else:
                    # Regular model call - create proper batch format
                    batch_dict = {'x': X_tensor, 'y': y_tensor}
                    if data_type == 'pk':
                        batch_dict = {'pk': batch_dict}
                    elif data_type == 'pd':
                        batch_dict = {'pd': batch_dict}
                    
                    results = self.model(batch_dict)
                    if data_type == 'pk':
                        predictions = results['pk']['pred'].cpu().numpy().flatten()
                    elif data_type == 'pd':
                        predictions = results['pd']['pred'].cpu().numpy().flatten()
        
        # Inverse transformation
        y_pred = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Create result dataframe
        results_df = test_data[['ID', 'TIME', 'DV']].copy()
        results_df['PRED'] = y_pred
        results_df['ERROR'] = y_pred - y_true
        results_df['ABS_ERROR'] = np.abs(results_df['ERROR'])
        
        print(f"    {data_type.upper()} prediction completed")
        
        return results_df
    
    def create_visualizations(self, pk_results, pd_results):
        """Creating visualizations"""
        print(" Creating visualizations...")
        
        # Create visualization directory
        viz_dir = self.run_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. PK prediction vs actual value
        self._plot_predictions_vs_actual(pk_results, "PK", viz_dir)
        
        # 2. PD prediction vs actual value
        self._plot_predictions_vs_actual(pd_results, "PD", viz_dir)
        
        # 3. PD value by ID and time
        self._plot_pd_by_id_time(pd_results, viz_dir)
        
        # 4. Error distribution
        self._plot_error_distribution(pk_results, pd_results, viz_dir)
        
        # 5. Performance metrics
        self._plot_performance_metrics(pk_results, pd_results, viz_dir)
        
        print(f"    Visualizations completed: {viz_dir}")
        
    def _plot_predictions_vs_actual(self, results, data_type, viz_dir):
        """Predictions vs actual values scatter plot"""
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(results['DV'], results['PRED'], alpha=0.6, s=50)
        
        # Perfect prediction line (y=x)
        min_val = min(results['DV'].min(), results['PRED'].min())
        max_val = max(results['DV'].max(), results['PRED'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # R² calculation
        from sklearn.metrics import r2_score
        r2 = r2_score(results['DV'], results['PRED'])
        
        plt.xlabel(f'Actual {data_type} Values')
        plt.ylabel(f'Predicted {data_type} Values')
        plt.title(f'{data_type} Predictions vs Actual Values\nR² = {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Ensure directory exists
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / f'{data_type.lower()}_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_pd_by_id_time(self, pd_results, viz_dir):
        """PD value by ID and time"""
        n_subj = pd_results['ID'].nunique()
        n_cols = 3
        n_rows = math.ceil(n_subj / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        axes = axes.flatten()

        for i, subject_id in enumerate(sorted(pd_results['ID'].unique())):
            subject_data = pd_results[pd_results['ID'] == subject_id].sort_values('TIME')

            ax = axes[i]
            ax.plot(subject_data['TIME'], subject_data['DV'], 'o-', label='Actual', linewidth=2, markersize=6)
            ax.plot(subject_data['TIME'], subject_data['PRED'], 's--', label='Predicted', linewidth=2, markersize=6)

            ax.set_title(f'Subject {subject_id}')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('PD Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 안 쓰는 subplot 비우기
        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.suptitle('PD Predictions by Subject and Time', fontsize=16)
        plt.tight_layout()

        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'pd_by_id_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_error_distribution(self, pk_results, pd_results, viz_dir):
        """오차 분포"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PK 오차 분포
        ax1.hist(pk_results['ERROR'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('PK Prediction Errors')
        ax1.set_xlabel('Error (Predicted - Actual)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # PD 오차 분포
        ax2.hist(pd_results['ERROR'], bins=30, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title('PD Prediction Errors')
        ax2.set_xlabel('Error (Predicted - Actual)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Ensure directory exists
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_performance_metrics(self, pk_results, pd_results, viz_dir):
        """성능 메트릭"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # 메트릭 계산
        pk_metrics = {
            'RMSE': np.sqrt(mean_squared_error(pk_results['DV'], pk_results['PRED'])),
            'MAE': mean_absolute_error(pk_results['DV'], pk_results['PRED']),
            'R²': r2_score(pk_results['DV'], pk_results['PRED'])
        }
        
        pd_metrics = {
            'RMSE': np.sqrt(mean_squared_error(pd_results['DV'], pd_results['PRED'])),
            'MAE': mean_absolute_error(pd_results['DV'], pd_results['PRED']),
            'R²': r2_score(pd_results['DV'], pd_results['PRED'])
        }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        metrics_names = list(pk_metrics.keys())
        metrics_values = list(pk_metrics.values())
        bars1 = ax1.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('PK Performance Metrics')
        ax1.set_ylabel('Value')
        for i, v in enumerate(metrics_values):
            ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        metrics_values = list(pd_metrics.values())
        bars2 = ax2.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('PD Performance Metrics')
        ax2.set_ylabel('Value')
        for i, v in enumerate(metrics_values):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        # Ensure directory exists
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 메트릭을 텍스트 파일로 저장
        with open(viz_dir / 'performance_metrics.txt', 'w') as f:
            f.write("=== Performance Metrics ===\n\n")
            f.write("PK Metrics:\n")
            for metric, value in pk_metrics.items():
                f.write(f"  {metric}: {value:.6f}\n")
            f.write("\nPD Metrics:\n")
            for metric, value in pd_metrics.items():
                f.write(f"  {metric}: {value:.6f}\n")
    
    def run_auto_visualization(self):
        """Running auto visualization"""
        print(" Auto visualization started!")
        
        try:
            # 1. Load experiment data
            self.load_experiment_data()
            
            # 2. 모델 로드
            self.load_model()
            
            # 3. Test 데이터 준비
            pk_test_data, pd_test_data, pk_features, pd_features = self.prepare_test_data()
            
            # 4. 예측 수행
            pk_results = self.make_predictions(
                pk_test_data, pk_features, 
                self.scalers['pk_target_scaler'], 'pk'
            )
            
            pd_results = self.make_predictions(
                pd_test_data, pd_features, 
                self.scalers['pd_target_scaler'], 'pd'
            )
            
            # 5. 시각화 생성
            self.create_visualizations(pk_results, pd_results)
            
            print(" Auto visualization completed!")
            
        except Exception as e:
            print(f" Error occurred: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto visualization script')
    parser.add_argument('--run_dir', type=str, required=True,
                       help='Experiment result directory path')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device setting (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # AutoVisualizer run
    visualizer = AutoVisualizer(args.run_dir, args.device)
    visualizer.run_auto_visualization()

if __name__ == "__main__":
    main()
