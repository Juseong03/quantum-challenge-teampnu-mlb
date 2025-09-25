#!/usr/bin/env python3
"""
Quantum Challenge - Model-Based PK/PD Solution
Population simulation for Tasks 1–5
"""

import torch
import numpy as np
import pandas as pd
import json, pickle
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt

from utils.factory import create_model
from utils.helpers import get_device
from config import Config
from data.loaders import features_from_dose_history

# ==========================================================
# Load model & scalers
# ==========================================================
model_path = Path("results/clf/dual_stage/mlp/s42")
config_path = model_path / "config.json"
scaler_path = model_path / "scalers.pkl"
features_path = model_path / "features.pkl"

print("🚀 MODEL-BASED PK/PD CHALLENGE SOLUTION")
print("=" * 60)

# Config
with open(config_path, "r") as f:
    config_dict = json.load(f)

config = Config()
for k, v in config_dict.items():
    if hasattr(config, k):
        setattr(config, k, v)

# Scalers
with open(scaler_path, "rb") as f:
    pk_scaler = pickle.load(f)
    pd_scaler = pickle.load(f)
    pk_target_scaler = pickle.load(f)
    pd_target_scaler = pickle.load(f)

# Features
with open(features_path, "rb") as f:
    feats = pickle.load(f)
pk_features = feats["pk"]
pd_features = feats["pd"]

# Device & Model
device = get_device()
model = create_model(config, None, pk_features, pd_features)
checkpoint = torch.load(model_path / "model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

print(f"✅ Model loaded: {config.mode} mode, {config.encoder} encoder")
print(f"📊 Input features: {len(pk_features)} features")

# ==========================================================
# Feature builder
# ==========================================================
def create_test_batch(dose_schedule, time, bw=75, comed=0, batch_size=1):
    """Build test batch consistent with training FE."""
    obs = pd.DataFrame([{
        "ID": 1, "TIME": time, "DV": 0.0, "DVID": 2,
        "BW": bw, "COMED": comed,
        "DOSE": dose_schedule[-1][1] if dose_schedule else 0.0
    }])
    dose = pd.DataFrame([{"ID": 1, "TIME": t, "AMT": a} for t, a in dose_schedule])

    fe_out = features_from_dose_history(
        obs, dose,
        add_pk_baseline=False,
        add_pd_delta=False,
        target="dv",
        allow_future_dose=False,
        add_decay_features=True,
        half_lives=[24, 48, 72],
        verbose=False
    )

    X = fe_out[pk_features].values
    X_scaled = pk_scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    return {
        "pk": {"x": X_tensor, "y": torch.zeros(batch_size, 1).to(device)},
        "pd": {"x": X_tensor, "y": torch.zeros(batch_size, 1).to(device)},
    }

# ==========================================================
# Population setup
# ==========================================================
df = pd.read_csv("data/EstData.csv")
subjects = df.groupby("ID")[["BW", "COMED"]].first().reset_index()
print(f"👥 Population size: {len(subjects)}")

baseline_threshold = 3.3

# ==========================================================
# Simulation function
# ==========================================================
def simulate_population(
    dose,
    schedule="daily",
    horizon=24,
    bw_shift=None,
    comed_allowed=True,
    return_details=False,
    verbose=False,
    use_clf=False,
    clf_threshold=0.5
):
    """Simulate biomarker suppression across population."""
    subject_results = {}

    for idx, row in subjects.iterrows():
        subj_id, bw, comed = row["ID"], row["BW"], row["COMED"]
        if not comed_allowed and comed == 1:
            continue
        if bw_shift == "wider":
            bw = np.interp(bw, [50, 100], [70, 140])

        if schedule == "daily":
            dose_schedule = [(24 * i, dose) for i in range(21)]
            start_time = 21 * 24
        elif schedule == "weekly":
            dose_schedule = [(168 * i, dose) for i in range(4)]
            start_time = 4 * 168
        else:
            raise ValueError("Unknown schedule")

        suppressed_all = True
        traj = []

        for t in range(horizon + 1):
            batch = create_test_batch(dose_schedule, time=start_time + t, bw=bw, comed=comed)
            with torch.no_grad():
                out = model(batch)
                pd_pred = out["pd"]["pred"].cpu().numpy().reshape(-1, 1)
                pd_pred = pd_target_scaler.inverse_transform(pd_pred)[0, 0]

                out_clf = model._forward_clf(batch)
                clf_prob = F.softmax(out_clf["pd"]["pred"], dim=1)[0, 1].item()

            traj.append({"time": start_time + t, "dv_pred": pd_pred, "prob_suppr": clf_prob})

            # suppression 판정
            if use_clf:
                if clf_prob <= clf_threshold:
                    suppressed_all = False
            else:
                if pd_pred >= baseline_threshold:
                    suppressed_all = False

        subject_results[subj_id] = {
            "suppressed": suppressed_all,
            "trajectory": traj if return_details else None
        }

    frac = np.mean([int(v["suppressed"]) for v in subject_results.values()])

    if verbose:
        print(f"✅ Dose {dose} mg → Suppression {frac:.1%}")

    return (frac, subject_results) if return_details else frac

# ==========================================================
# Utility: dose–suppression plot
# ==========================================================
def plot_dose_response(doses, fracs, title="Suppression Curve"):
    plt.figure(figsize=(7,5))
    plt.plot(doses, fracs, marker="o", label="Suppression fraction")
    plt.axhline(0.9, color="red", ls="--", label="90% target")
    plt.axhline(0.75, color="orange", ls="--", label="75% target")
    plt.xlabel("Dose (mg)")
    plt.ylabel("Suppression fraction")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================================
# Tasks
# ==========================================================
print("\n🎯 TASK 1: Daily dosing (0.5 mg step, boundary 0.5–25 mg)")
doses = np.arange(0.5, 15.0, 0.1)
fracs = []
best_dose, best_frac = None, 0
for dose in doses:
    frac = simulate_population(dose, schedule="daily", horizon=24, verbose=True)
    fracs.append(frac)
    if best_dose is None and frac >= 0.9:
        best_dose, best_frac = dose, frac
print(f"  ✅ Optimal daily dose: {best_dose} mg (Suppression {best_frac:.1%})")
plot_dose_response(doses, fracs, "Daily dosing suppression")

print("\n🎯 TASK 2: Weekly dosing (5 mg step)")
doses = np.arange(5, 55, 1)
fracs = []
best_dose, best_frac = None, 0
for dose in doses:
    frac = simulate_population(dose, schedule="weekly", horizon=168)
    fracs.append(frac)
    if best_dose is None and frac >= 0.9:
        best_dose, best_frac = dose, frac
print(f"  ✅ Optimal weekly dose: {best_dose} mg (Suppression {best_frac:.1%})")
plot_dose_response(doses, fracs, "Weekly dosing suppression")

print("\n🎯 TASK 3: BW shifted 70–140 kg")
daily_dose = simulate_population(10, schedule="daily", bw_shift="wider")
weekly_dose = simulate_population(20, schedule="weekly", bw_shift="wider")
print(f"  Daily suppression at 10 mg: {daily_dose:.1%}")
print(f"  Weekly suppression at 20 mg: {weekly_dose:.1%}")

print("\n🎯 TASK 4: No concomitant medication allowed")
daily_dose = simulate_population(10, schedule="daily", comed_allowed=False)
weekly_dose = simulate_population(20, schedule="weekly", comed_allowed=False)
print(f"  Daily suppression at 10 mg: {daily_dose:.1%}")
print(f"  Weekly suppression at 20 mg: {weekly_dose:.1%}")

print("\n🎯 TASK 5: 75% suppression criterion")
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
