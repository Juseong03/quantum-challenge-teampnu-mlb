# PK/PD Modeling System

A comprehensive machine learning system for Pharmacokinetic (PK) and Pharmacodynamic (PD) modeling with multiple training modes and advanced features.

## 🚀 **Quick Start**

### **Training a Model**
```bash
# Basic training (now with deterministic splitting by default)
python main.py --mode separate --encoder mlp --epochs 100

# Training with feature engineering
python main.py --mode separate --encoder mlp --epochs 100 --use_fe
```

### **Making Predictions**
```bash
# Basic prediction (now with improved consistency by default)
python predict.py \
    --model_path results/runs/your_run/separate/mlp/s42 \
    --data_path data/new_data.csv \
    --output_path predictions.csv

# Prediction with uncertainty quantification
python predict.py \
    --model_path results/runs/your_run/separate/mlp/s42 \
    --data_path data/new_data.csv \
    --output_path predictions_with_uncertainty.csv \
    --uncertainty
```

## 📁 **Project Structure**

```
TeamPNU2/
├── main.py                    # Main training script (with deterministic splitting)
├── predict.py                 # Prediction script (with scaler consistency)
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
│
├── data/                      # Data handling
│   ├── loaders.py            # Data loading utilities
│   ├── splits.py             # Data splitting strategies
│   └── EstData.csv           # Sample dataset
│
├── models/                    # Model architectures
│   ├── unified_model.py      # Unified PK/PD model
│   ├── encoders.py           # Encoder implementations
│   └── heads.py              # Prediction heads
│
├── training/                  # Training utilities
│   └── unified_trainer.py    # Unified trainer for all modes
│
├── utils/                     # Utility functions
│   ├── helpers.py            # Helper functions
│   ├── factory.py            # Model/trainer factory
│   └── logging.py            # Logging utilities
│
├── scripts/                   # Analysis and debugging scripts
│   ├── fix_scaler_issue.py   # Scaler consistency fixes
│   ├── test_prediction.py    # Prediction testing
│   └── compare_training_inference.py  # Training vs inference comparison
│
├── analysis_tools/            # Analysis and visualization tools
│   ├── performance_comparison_analysis.py
│   ├── pd_focused_analysis.py
│   └── interactive_analysis_dashboard.py
│
├── docs/                      # Documentation and reports
│   ├── PREDICTION_GUIDE.md   # Detailed prediction guide
│   ├── ADDING_NEW_ENCODER.md # Guide for adding new encoders
│   └── *.png                 # Analysis plots and visualizations
│
└── results/                   # Training results and models
    ├── runs/                 # Hierarchical result storage
    ├── models/               # Model symlinks
    └── logs/                 # Training logs
```

## 🎯 **Key Features**

### **Training Modes**
- **Separate**: PK and PD models trained independently
- **Joint**: PK and PD trained together with shared information
- **Dual Stage**: PK first, then PD with PK information
- **Integrated**: PK encoder output used as PD encoder input
- **Shared**: Common encoder for both PK and PD
- **Two Stage Shared**: One shared encoder used in two stages

### **Model Architectures**
- **MLP**: Multi-layer perceptron
- **ResMLP**: Residual MLP
- **MoE**: Mixture of Experts
- **ResMLP-MoE**: Residual MLP with MoE
- **Adaptive ResMLP-MoE**: Adaptive version
- **CNN**: Convolutional neural network

### **Advanced Features**
- **Feature Engineering**: Time windows, per-kg dosing, future dose information
- **Data Augmentation**: Mixup augmentation
- **PK/PD Contrastive Learning**: Domain-specific positive/negative pair generation
  - Temporal continuity (same patient, consecutive time points)
  - Dose group similarity (same dose group patients)
  - Feature similarity (data-driven similarity)
  - Hybrid strategy (combination of above)
- **Uncertainty Quantification**: Monte Carlo Dropout
- **Deterministic Splitting**: Consistent data splits across runs

## 🔧 **Configuration**

The system uses a comprehensive configuration system. Key parameters:

```python
# Training settings
epochs: int = 3000
batch_size: int = 32
learning_rate: float = 1e-3
patience: int = 300

# Model settings
encoder: str = "mlp"  # or "resmlp", "moe", etc.
mode: str = "separate"  # or "joint", "dual_stage", etc.

# Data settings
use_feature_engineering: bool = False
use_mixup: bool = False
use_mc_dropout: bool = False
```

## 📊 **Data Format**

Input CSV should have the following columns:
- `ID`: Subject identifier
- `TIME`: Time point
- `DV`: Target variable (concentration/effect)
- `DVID`: Data type (1 for PK, 2 for PD)
- `BW`: Body weight
- `COMED`: Concomitant medication
- `DOSE`: Dose amount
- `EVID`: Event identifier
- `MDV`: Missing dependent variable
- `AMT`: Amount
- `CMT`: Compartment

## 🎯 **Prediction Output**

The prediction scripts generate CSV files with:
- `PK_PREDICTION`: Predicted PK values
- `PD_PREDICTION`: Predicted PD values
- `PK_STD`, `PD_STD`: Uncertainty estimates (if --uncertainty used)
- `PK_CI_LOWER`, `PK_CI_UPPER`: Confidence intervals
- `PD_CI_LOWER`, `PD_CI_UPPER`: Confidence intervals

## 🔍 **Scaler Consistency**

The system includes advanced scaler consistency features by default:

- **Deterministic Data Splitting**: Ensures consistent data splits across runs
- **Saved Scaler Usage**: Uses exact scalers from training for inference
- **Consistent Preprocessing**: Same preprocessing pipeline for training and inference

## 📈 **Performance Monitoring**

Training results are saved with comprehensive metrics:
- RMSE, MAE, R² for both PK and PD
- Training/validation loss curves
- Model parameters and architecture info
- Split information for reproducibility

## 🛠️ **Installation**

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic training
python main.py --mode separate --encoder mlp --epochs 100

# Test prediction functionality
python scripts/test_prediction.py
```

## 📚 **Documentation**

- **[Prediction Guide](docs/PREDICTION_GUIDE.md)**: Detailed guide for making predictions
- **[Adding New Encoders](docs/ADDING_NEW_ENCODER.md)**: Guide for extending the system
- **[Analysis Reports](docs/)**: Various analysis reports and visualizations

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Data Format Issues**: Ensure CSV has required columns
2. **CUDA Memory**: Use `--device cpu` for CPU inference
3. **Feature Mismatch**: Check if feature engineering was used during training
4. **Model Not Found**: Check model path and directory structure

### **Debugging Tools**

- `scripts/test_prediction.py`: Test prediction functionality
- `scripts/compare_training_inference.py`: Compare training vs inference results
- `scripts/fix_scaler_issue.py`: Scaler consistency analysis tools

## 🎯 **Best Practices**

1. **Use Default Scripts**: `main.py` and `predict.py` now include all improvements by default
2. **Save Split Information**: Always save split info for reproducibility
3. **Monitor Training**: Check logs and results for training progress
4. **Validate Predictions**: Test predictions on validation data
5. **Check Data Format**: Ensure input data matches training data format

## 📊 **Example Workflow**

```bash
# 1. Train a model (now with deterministic splitting by default)
python main.py --mode separate --encoder mlp --epochs 100

# 2. Test prediction functionality
python scripts/test_prediction.py

# 3. Make predictions on new data (now with improved consistency by default)
python predict.py \
    --model_path results/runs/your_run/separate/mlp/s42 \
    --data_path data/new_data.csv \
    --output_path predictions.csv

# 4. Analyze results
python analysis_tools/performance_comparison_analysis.py
```

## 🤝 **Contributing**

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use deterministic splitting for consistency

## 📄 **License**

This project is part of the TeamPNU research initiative.

---

**For detailed information, see the documentation in the `docs/` directory.**
