#!/usr/bin/env python3
"""
Quantum Challenge - Model-Based PK/PD Solution
Population simulation for Tasks 1–5
"""

import torch
import numpy as np
import pandas as pd
import json, pickle
from pathlib import Path
from utils.factory import create_model
from utils.helpers import get_device
from config import Config

# ==========================================================
# Ensemble loader
# ==========================================================
def load_model_bundle(model_dir: Path):
    """하나의 학습 결과 디렉토리에서 모델+스케일러+피처 불러오기"""
    with open(model_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # scalers
    with open(model_dir / "scalers.pkl", "rb") as f:
        pk_scaler = pickle.load(f)
        pd_scaler = pickle.load(f)
        pk_target_scaler = pickle.load(f)
        pd_target_scaler = pickle.load(f)

    # features
    with open(model_dir / "features.pkl", "rb") as f:
        feats = pickle.load(f)
    pk_features = feats["pk"]
    pd_features = feats["pd"]

    # model
    model = create_model(config, None, pk_features, pd_features)
    checkpoint = torch.load(model_dir / "model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return {
        "model": model,
        "pk_scaler": pk_scaler,
        "pd_scaler": pd_scaler,
        "pk_target_scaler": pk_target_scaler,
        "pd_target_scaler": pd_target_scaler,
        "pk_features": pk_features,
        "pd_features": pd_features,
    }


from data.loaders import features_from_dose_history, use_feature_engineering  # <-- FE 코드 import  

# ==========================================================
# Ensemble setup
# ==========================================================
device = get_device()
model_paths = [
    Path("results/baseline/dual_stage/mlp/s1"),
    Path("results/baseline/dual_stage/mlp/s2"),
    Path("results/baseline/dual_stage/mlp/s3"),
    Path("results/baseline2/dual_stage/mlp/s1"),
    Path("results/baseline2/dual_stage/mlp/s2"),
    Path("results/baseline2/dual_stage/mlp/s3"),
]
bundles = [load_model_bundle(p) for p in model_paths]
print(f"✅ Loaded {len(bundles)} models for ensemble")

# ==========================================================
# Ensemble prediction
# ==========================================================
def create_test_batch(dose_schedule, time, bw, comed, bundle, batch_size=1):
    """특정 모델 bundle에 맞는 feature vector 생성"""
    obs = pd.DataFrame([{
        "ID": 1, "TIME": time, "DV": 0.0, "DVID": 2,
        "BW": bw, "COMED": comed,
        "DOSE": dose_schedule[-1][1] if dose_schedule else 0.0
    }])
    dose = pd.DataFrame([{"ID": 1, "TIME": t, "AMT": a} for t, a in dose_schedule])

    fe_out = features_from_dose_history(
        obs, dose,
        add_pk_baseline=False, add_pd_delta=False,
        target="dv", allow_future_dose=False,
        time_windows=None, add_decay_features=True,
        half_lives=[24, 48, 72]
    )

    X = fe_out[bundle["pk_features"]].values
    X_scaled = bundle["pk_scaler"].transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    return {
        "pk": {"x": X_tensor, "y": torch.zeros(batch_size, 1).to(device)},
        "pd": {"x": X_tensor, "y": torch.zeros(batch_size, 1).to(device)},
    }

def ensemble_predict(dose_schedule, time, bw, comed):
    """여러 모델의 PD 예측 평균"""
    preds = []
    for bundle in bundles:
        batch = create_test_batch(dose_schedule, time, bw, comed, bundle)
        with torch.no_grad():
            pred = bundle["model"](batch)["pd"]["pred"]
            biom = bundle["pd_target_scaler"].inverse_transform(
                pred.cpu().numpy().reshape(-1, 1)
            )[0, 0]
        preds.append(biom)
    return np.mean(preds)


# ==========================================================
# Population setup
# ==========================================================
df = pd.read_csv("data/EstData.csv")
subjects = df.groupby("ID")[["BW","COMED"]].first().reset_index()
print(f"👥 Population size: {len(subjects)}")

baseline_threshold = 3.3

# ==========================================================
# Simulation with ensemble
# ==========================================================
def simulate_population(dose, schedule="daily", horizon=24, bw_shift=None, comed_allowed=True):
    suppress_flags = []
    for _, row in subjects.iterrows():
        bw, comed = row["BW"], row["COMED"]
        if not comed_allowed and comed == 1:
            continue
        if bw_shift == "wider":
            bw = np.interp(bw, [50, 100], [70, 140])

        if schedule == "daily":
            dose_schedule = [(24*i, dose) for i in range(21)]
            start_time = 21*24
        elif schedule == "weekly":
            dose_schedule = [(168*i, dose) for i in range(4)]
            start_time = 4*168
        else:
            raise ValueError("Unknown schedule")

        biomarker_vals = []
        for t in range(horizon+1):
            biom = ensemble_predict(dose_schedule, start_time+t, bw, comed)
            biomarker_vals.append(biom)

        min_val = min(biomarker_vals)
        suppress_flags.append(min_val < baseline_threshold)
    return np.mean(suppress_flags)

# ==========================================================
# Task 1 & 2: optimal doses
# ==========================================================
print("\n🎯 TASK 1: Daily dosing (0.5mg step)")
best_dose, best_frac = None, 0
for dose in np.arange(1, 15, 0.5):
    frac = simulate_population(dose, schedule="daily", horizon=24)
    if frac >= 0.9:
        best_dose, best_frac = dose, frac
        break
print(f"  ✅ Optimal daily dose: {best_dose} mg (Suppression {best_frac:.1%})")

print("\n🎯 TASK 2: Weekly dosing (5mg step)")
best_dose, best_frac = None, 0
for dose in np.arange(10, 55, 5):
    frac = simulate_population(dose, schedule="weekly", horizon=168)
    if frac >= 0.9:
        best_dose, best_frac = dose, frac
        break
print(f"  ✅ Optimal weekly dose: {best_dose} mg (Suppression {best_frac:.1%})")

# ==========================================================
# Task 3: BW distribution shift
# ==========================================================
print("\n🎯 TASK 3: BW shifted 70–140 kg")
daily_dose = simulate_population(10, schedule="daily", bw_shift="wider")
weekly_dose = simulate_population(20, schedule="weekly", bw_shift="wider")
print(f"  Daily example suppression at 10mg: {daily_dose:.1%}")
print(f"  Weekly example suppression at 20mg: {weekly_dose:.1%}")

# ==========================================================
# Task 4: No concomitant meds
# ==========================================================
print("\n🎯 TASK 4: No concomitant medication allowed")
daily_dose = simulate_population(10, schedule="daily", comed_allowed=False)
weekly_dose = simulate_population(20, schedule="weekly", comed_allowed=False)
print(f"  Daily example suppression at 10mg: {daily_dose:.1%}")
print(f"  Weekly example suppression at 20mg: {weekly_dose:.1%}")

# ==========================================================
# Task 5: 75% suppression criterion
# ==========================================================
print("\n🎯 TASK 5: 75% suppression instead of 90%")
best_dose, best_frac = None, 0
for dose in np.arange(0.5, 25.5, 0.5):
    frac = simulate_population(dose, schedule="daily", horizon=24)
    if frac >= 0.75:
        best_dose, best_frac = dose, frac
        break
print(f"  ✅ Daily dose (75% criterion): {best_dose} mg")

best_dose, best_frac = None, 0
for dose in np.arange(5, 55, 5):
    frac = simulate_population(dose, schedule="weekly", horizon=168)
    if frac >= 0.75:
        best_dose, best_frac = dose, frac
        break
print(f"  ✅ Weekly dose (75% criterion): {best_dose} mg")

print("\n✅ Challenge solution completed!")
