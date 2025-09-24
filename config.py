"""
 Configuration
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """Configuration"""
    
    # === Basic settings ===
    mode: str = "separate"  # "separate", "joint", "shared", "dual_stage", "integrated", "two_stage_shared"
    csv_path: str = "data/EstData.csv"
    
    # === Training settings ===
    epochs: int = 1000  # Optimized for TIME feature enhancement model
    batch_size: int = 16  # Increased for better GPU utilization and stability
    learning_rate: float = 1e-3  # Optimized for TIME feature enhancement convergence
    patience: int = 100  # Balanced early stopping for TIME feature model
    
    # === Model settings ===
    encoder: str = "mlp"  # "mlp", "resmlp", "moe", "resmlp_moe", "hybrid_mlp", "hybrid_resmlp", "hybrid_moe", "hybrid_resmlp_moe", "qmlp", "qresmlp", "qmoe", "qresmlp_moe"
    encoder_pk: Optional[str] = None  # PK-specific encoder
    encoder_pd: Optional[str] = None  # PD-specific encoder
    head_pk: str = "mse"  # "mse", "gauss", "poisson"
    head_pd: str = "mse"  # "mse", "gauss", "poisson", "emax"
    
    # === Model hyperparameters ===
    hidden: int = 128  # Increased for TIME feature enhancement model capacity
    depth: int = 4  # Increased depth for better TIME pattern learning
    dropout: float = 0.3  # Increased for better regularization with TIME features
    
    # === MoE settings ===
    num_experts: int = 8  # Increased for better expert diversity
    top_k: int = 4  # Optimal for most cases
    
    # === CNN settings ===
    kernel_size: int = 3  # Good for time series patterns
    num_filters: int = 128  # Increased for better feature extraction
    
    # === QNN settings ===
    n_qubits: int = 6  # Increased for better quantum capacity
    qnn_layers: int = 3  # Increased for more complex quantum circuits
    quantum_ratio: float = 0.3  # Reduced for more stable hybrid performance
    use_entanglement: str = "linear"  # "linear", "circular", "all"
    use_data_reuploading: bool = True
    quantum_frequency: int = 2  # For quantum_resmlp
    
    # === Uncertainty Quantification ===
    use_mc_dropout: bool = False  # Disabled by default for faster training
    mc_dropout_rate: float = 0.2  # Matches main dropout rate
    mc_samples: int = 100  # Increased for better uncertainty estimation
    
    # === Data preprocessing ===
    use_feature_engineering: bool = True  # Enabled by default for better performance
    perkg: bool = False
    allow_future_dose: bool = False
    time_windows: List[int] = None
    
    # === Data augmentation ===
    aug_method: str = None  # "mixup", "jitter", "jitter_mixup", "gaussian_noise", "scaling", "time_warp", "feature_dropout", "cutmix", "random_erase", "label_smooth", "amplitude_scale", "enhanced_mixup", "random", "pk_curve", "pd_response", or None
    aug_ratio: float = None  # Ratio of augmented samples to original data (e.g., 0.2 for 20%)
    aug_samples: int = 100  # Number of augmentation samples (used if aug_ratio is None)
    mixup_alpha: float = 0.3  # Mixup alpha parameter
    jitter_std: float = 0.05  # Standard deviation for DV jitter
    time_shift_ratio: float = 0.1  # Ratio for TIME shift
    
    # Enhanced augmentation parameters
    gaussian_noise_std: float = 0.02  # Standard deviation for Gaussian noise
    scale_range: tuple = (0.8, 1.2)  # Range for random scaling
    dropout_rate: float = 0.1  # Dropout rate for augmentation
    time_warp_factor: float = 0.1  # Factor for time warping
    amplitude_scale_range: tuple = (0.9, 1.1)  # Range for amplitude scaling
    cutmix_alpha: float = 1.0  # Alpha parameter for CutMix
    cutmix_prob: float = 0.1  # Probability for CutMix
    label_smooth_eps: float = 0.1  # Epsilon for label smoothing
    random_erase_prob: float = 0.1  # Probability for random erasing
    feature_dropout_prob: float = 0.1  # Probability for feature dropout

    # Supervised training augmentation
    use_aug_supervised: bool = False  # Use augmentation during supervised training
    aug_lambda: float = 0.3  # Weight for augmented loss (original + lambda * augmented)

    # Legacy mixup settings (for backward compatibility)
    use_mixup: bool = False  # Legacy: use aug_method instead
    mixup_prob: float = 0.1  # Legacy: use aug_method instead
    
    # === SimCLR Contrastive learning ===
    temperature: float = 0.1  # Temperature for SimCLR contrastive learning
    time_jitter_std: float = 0.1  # Time jitter standard deviation
    noise_std: float = 0.05  # Gaussian noise standard deviation
    contrastive_scale_range: tuple = (0.8, 1.2)  # Scaling range for contrastive augmentation
    pretraining_epochs: int = 50  # Number of pretraining epochs
    pretraining_patience: int = 20  # Patience for contrastive pretraining
    use_contrastive_pretraining: bool = False  # Enable contrastive pretraining (set by --use_contrastive_pretraining flag)

    # === Data splitting ===
    split_strategy: str = "stratify_dose_even"  # Best strategy for PK/PD data
    test_size: float = 0.1  # Increased for better test evaluation
    val_size: float = 0.1  # Increased for better validation
    random_state: int = 42  # Reproducible results
    
    # === Output settings ===
    output_dir: str = "results"
    run_name: Optional[str] = None
    verbose: bool = False
    device_id: int = 0
    
    def __post_init__(self):
        """Initialization after processing"""
        if self.time_windows is None:
            self.time_windows = [24, 48, 72, 96, 120, 144, 168]


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="PK/PD Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # === Basic settings ===
    parser.add_argument("--mode", 
                       choices=["separate", "joint", "shared", "dual_stage", "integrated", "two_stage_shared"], 
                       default="separate", help="Training mode")
    parser.add_argument("--csv", default="data/EstData.csv", help="Data CSV file path")
    
    # === Training settings ===
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    
    # === Model settings ===
    parser.add_argument("--encoder", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "cnn", "qmlp", "qresmlp", "qmoe", "qresmlp_moe",
                                "qnn", "resqnn_moe",
                                ],  
                       default="mlp", help="Default encoder type")
    parser.add_argument("--encoder_pk", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "cnn", "qmlp", "qresmlp", "qmoe", "qresmlp_moe"], 
                       default=None, help="PK-specific encoder type")
    parser.add_argument("--encoder_pd", 
                       choices=["mlp", "resmlp", "moe", "resmlp_moe", "cnn", "qmlp", "qresmlp", "qmoe", "qresmlp_moe"], 
                       default=None, help="PD-specific encoder type")

    parser.add_argument("--head_pk", choices=["mse", "gauss", "poisson"], 
                       default="mse", help="PK head type")
    parser.add_argument("--head_pd", choices=["mse", "gauss", "poisson", "emax"], 
                       default="mse", help="PD head type")
    
    # === Model hyperparameters ===
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=4, help="Network depth")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout ratio")
    
    # === MoE settings ===
    parser.add_argument("--num_experts", type=int, default=8, help="Number of MoE experts")
    parser.add_argument("--top_k", type=int, default=4, help="Number of top experts to use")
    
    # === CNN settings ===
    parser.add_argument("--kernel_size", type=int, default=3, help="CNN kernel size")
    parser.add_argument("--num_filters", type=int, default=128, help="Number of CNN filters")
    
    # === QNN settings ===
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits for QNN")
    parser.add_argument("--qnn_layers", type=int, default=2, help="Number of quantum layers")
    parser.add_argument("--quantum_ratio", type=float, default=0.3, help="Quantum ratio for hybrid QNN")
    parser.add_argument("--use_entanglement", choices=["linear", "circular", "all"], 
                       default="linear", help="Entanglement pattern for QNN")
    parser.add_argument("--use_data_reuploading", action="store_true", help="Use data reuploading in QNN")
    parser.add_argument("--quantum_frequency", type=int, default=2, help="Quantum frequency for quantum_resmlp")
    
    # === Data preprocessing ===
    parser.add_argument("--use_fe", action="store_true", help="Feature engineering")
    parser.add_argument("--perkg", action="store_true", help="Per kg dose")
    parser.add_argument("--allow_future_dose", action="store_true", help="Allow future dose information")
    parser.add_argument("--time_windows", type=str, default=None,
                       help="Time windows (comma-separated, e.g., '24,48,72,96,120,144,168')")
    
    # === Data augmentation ===
    parser.add_argument("--aug_method", choices=[
        "mixup", "jitter", "jitter_mixup", "gaussian_noise", "scaling",
        "time_warp", "feature_dropout", "cutmix", "random_erase",
        "label_smooth", "amplitude_scale", "enhanced_mixup",
        "random", "pk_curve", "pd_response"
    ], help="Augmentation method")
    parser.add_argument("--aug_ratio", type=float, default=None,
                       help="Ratio of augmented samples to original data (e.g., 0.2 for 20%)")
    parser.add_argument("--aug_samples", type=int, default=100,
                       help="Number of augmentation samples (used if aug_ratio is None)")
    parser.add_argument("--mixup_alpha", type=float, default=0.3,
                       help="Mixup alpha parameter (default: 0.3)")
    parser.add_argument("--jitter_std", type=float, default=0.05,
                       help="Standard deviation for DV jitter (default: 0.05)")
    parser.add_argument("--time_shift_ratio", type=float, default=0.1,
                       help="Ratio for TIME shift (default: 0.1)")

    # Enhanced augmentation parameters
    parser.add_argument("--gaussian_noise_std", type=float, default=0.02,
                       help="Standard deviation for Gaussian noise (default: 0.02)")
    parser.add_argument("--scale_range", nargs=2, type=float, default=[0.8, 1.2],
                       help="Range for random scaling (default: [0.8, 1.2])")
    parser.add_argument("--time_warp_factor", type=float, default=0.1,
                       help="Factor for time warping (default: 0.1)")
    parser.add_argument("--amplitude_scale_range", nargs=2, type=float, default=[0.9, 1.1],
                       help="Range for amplitude scaling (default: [0.9, 1.1])")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                       help="Alpha parameter for CutMix (default: 1.0)")
    parser.add_argument("--cutmix_prob", type=float, default=0.1,
                       help="Probability for CutMix (default: 0.1)")
    parser.add_argument("--label_smooth_eps", type=float, default=0.1,
                       help="Epsilon for label smoothing (default: 0.1)")
    parser.add_argument("--random_erase_prob", type=float, default=0.1,
                       help="Probability for random erasing (default: 0.1)")
    parser.add_argument("--feature_dropout_prob", type=float, default=0.1,
                       help="Probability for feature dropout (default: 0.1)")
    parser.add_argument("--use_aug_supervised", action="store_true",
                       help="Use augmentation during supervised training")
    parser.add_argument("--aug_lambda", type=float, default=0.3,
                       help="Weight for augmented loss (original + lambda * augmented)")

    # === SimCLR Contrastive learning ===
    parser.add_argument("--temperature", type=float, default=0.1, help="SimCLR temperature")
    parser.add_argument("--time_jitter_std", type=float, default=0.1, help="Time jitter standard deviation")
    parser.add_argument("--noise_std", type=float, default=0.05, help="Gaussian noise standard deviation")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for augmentation")
    parser.add_argument("--contrastive_scale_range", nargs=2, type=float, default=[0.8, 1.2], help="Scaling range for contrastive augmentation")
    parser.add_argument("--pretraining_epochs", type=int, default=50, help="Number of contrastive pretraining epochs")
    parser.add_argument("--pretraining_patience", type=int, default=20, help="Patience for contrastive pretraining")
    parser.add_argument("--use_contrastive_pretraining", action="store_true", help="Enable contrastive pretraining")
    
    # === Data splitting ===
    parser.add_argument("--split_strategy", default="stratify_dose_even_no_placebo_valtest", # "stratify_dose_even", "stratify_dose_even_no_placebo_test", "leave_one_dose_out", "random_subject", "only_bw_range" "highest_bw_one_test", "stratify_dose_even_no_placebo_valtest"
                       help="Data splitting strategy")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set size")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # === Uncertainty Quantification ===
    parser.add_argument("--use_mc_dropout", action="store_true", help="Use Monte Carlo Dropout for uncertainty quantification")
    parser.add_argument("--mc_dropout_rate", type=float, default=0.15, help="Dropout rate for MC Dropout")
    parser.add_argument("--mc_samples", type=int, default=50, help="Number of MC samples for uncertainty estimation")
    
    # === Output settings ===
    parser.add_argument("--out_dir", default="results", help="Result output directory")
    parser.add_argument("--run_name", help="Run name (auto-generated)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device ID")
    
    return parser


def parse_time_windows(time_windows_str: str) -> List[int]:
    """Parse time windows string"""
    if not time_windows_str:
        return None
    try:
        return [int(x.strip()) for x in time_windows_str.split(',') if x.strip()]
    except ValueError:
        raise ValueError(f"Invalid time windows format: {time_windows_str}. Use comma-separated integers (e.g., '24,48,72,96,120,144,168')")


def parse_args() -> Config:
    """Parse command line arguments and create Config object"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create Config object
    config = Config(
        # Basic settings
        mode=args.mode,
        csv_path=args.csv,
        
        # Training settings
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        
        # Model settings
        encoder=args.encoder,
        encoder_pk=args.encoder_pk,
        encoder_pd=args.encoder_pd,
        head_pk=args.head_pk,
        head_pd=args.head_pd,
        
        # Model hyperparameters
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        
        # MoE settings
        num_experts=args.num_experts,
        top_k=args.top_k,
        
        # CNN settings
        kernel_size=args.kernel_size,
        num_filters=args.num_filters,
        
        # QNN settings
        n_qubits=args.n_qubits,
        qnn_layers=args.qnn_layers,
        quantum_ratio=args.quantum_ratio,
        use_entanglement=args.use_entanglement,
        use_data_reuploading=args.use_data_reuploading,
        quantum_frequency=args.quantum_frequency,
        
        # Uncertainty Quantification
        use_mc_dropout=args.use_mc_dropout,
        mc_dropout_rate=args.mc_dropout_rate,
        mc_samples=args.mc_samples,
        
        # Data preprocessing
        use_feature_engineering=args.use_fe,
        perkg=args.perkg,
        allow_future_dose=args.allow_future_dose,
        time_windows=parse_time_windows(args.time_windows) if args.time_windows else None,
        
        # Data augmentation
        aug_method=args.aug_method,
        aug_ratio=args.aug_ratio,
        aug_samples=args.aug_samples,
        mixup_alpha=args.mixup_alpha,
        jitter_std=args.jitter_std,
        time_shift_ratio=args.time_shift_ratio,

        # Enhanced augmentation parameters
        gaussian_noise_std=args.gaussian_noise_std,
        contrastive_scale_range=tuple(args.contrastive_scale_range),
        dropout_rate=args.dropout_rate,
        time_warp_factor=args.time_warp_factor,
        amplitude_scale_range=tuple(args.amplitude_scale_range),
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        label_smooth_eps=args.label_smooth_eps,
        random_erase_prob=args.random_erase_prob,
        feature_dropout_prob=args.feature_dropout_prob,

        # Supervised training augmentation
        use_aug_supervised=args.use_aug_supervised,
        aug_lambda=args.aug_lambda,

        # Legacy mixup settings
        use_mixup=False,
        mixup_prob=0.1,

        temperature=args.temperature,
        time_jitter_std=args.time_jitter_std,   
        noise_std=args.noise_std,
        pretraining_epochs=args.pretraining_epochs,
        pretraining_patience=args.pretraining_patience,
        use_contrastive_pretraining=args.use_contrastive_pretraining,
        
        # Data splitting
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        
        # Output settings
        output_dir=args.out_dir,
        run_name=args.run_name,
        verbose=args.verbose,
        device_id=args.device_id
    )
    
    return config