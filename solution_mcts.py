#!/usr/bin/env python3
"""
Accurate MCTS Optimization for Quantum Challenge Tasks 1–5
with Ensemble + Regression/Classification Suppression Criteria
"""

import numpy as np
import math, random
import torch
import pandas as pd
import json, pickle
from pathlib import Path
import torch.nn.functional as F

from utils.factory import create_model
from utils.helpers import get_device
from config import Config
from data.loaders import features_from_dose_history

# ==========================================================
# Ensemble loader
# ==========================================================
def load_model_bundle(model_dir: Path, device):
    """Load model + scalers + features from one training run"""
    with open(model_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    with open(model_dir / "scalers.pkl", "rb") as f:
        pk_scaler = pickle.load(f)
        pd_scaler = pickle.load(f)
        pk_target_scaler = pickle.load(f)
        pd_target_scaler = pickle.load(f)

    with open(model_dir / "features.pkl", "rb") as f:
        feats = pickle.load(f)
    pk_features = feats["pk"]
    pd_features = feats["pd"]

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

# ==========================================================
# Ensemble setup
# ==========================================================
device = get_device()
model_paths = [
    Path("results/clf/dual_stage/mlp/s42"),
]
bundles = [load_model_bundle(p, device) for p in model_paths]
print(f"✅ Loaded {len(bundles)} models for ensemble")

baseline_threshold = 3.3

# ==========================================================
# Ensemble prediction
# ==========================================================
def create_test_batch(dose_schedule, time, bw, comed, bundle, batch_size=1):
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
    """Return ensemble mean regression + classification"""
    reg_preds, clf_probs = [], []
    for bundle in bundles:
        batch = create_test_batch(dose_schedule, time, bw, comed, bundle)
        with torch.no_grad():
            out = bundle["model"](batch)
            reg_val = out["pd"]["pred"].cpu().numpy().reshape(-1, 1)
            reg_val = bundle["pd_target_scaler"].inverse_transform(reg_val)[0, 0]
            reg_preds.append(reg_val)

            out_clf = bundle["model"]._forward_clf(batch)["pd"]["pred"]
            prob = F.softmax(out_clf, dim=1)[0, 1].item()
            clf_probs.append(prob)
    return np.mean(reg_preds), np.mean(clf_probs)

# ==========================================================
# Population setup
# ==========================================================
df = pd.read_csv("data/EstData.csv")
subjects = df.groupby("ID")[["BW","COMED"]].first().reset_index()
print(f"👥 Population size: {len(subjects)}")

# ==========================================================
# Accurate population simulation (full horizon, all subjects)
# ==========================================================
def simulate_population(
    dose,
    schedule="daily",
    horizon=24,
    bw_shift=None,
    comed_allowed=True,
    use_clf=False,
    strict=True
):
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

        biom_vals, prob_vals = [], []
        for t in range(horizon+1):  # full horizon
            biom, prob = ensemble_predict(dose_schedule, start_time+t, bw, comed)
            biom_vals.append(biom)
            prob_vals.append(prob)

        if use_clf:
            suppr = np.mean(prob_vals) > 0.5
        else:
            if strict:
                suppr = max(biom_vals) < baseline_threshold
            else:
                suppr = min(biom_vals) < baseline_threshold

        suppress_flags.append(suppr)

    return np.mean(suppress_flags)

# ==========================================================
# MCTS Node
# ==========================================================
class MCTSNode:
    def __init__(self, dose, parent=None):
        self.dose = dose
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.reward = 0.0

    def ucb_score(self, exploration=1.4):
        if self.visits == 0:
            return float("inf")
        return (self.reward / self.visits) + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self, exploration=1.4):
        return max(self.children.values(), key=lambda c: c.ucb_score(exploration))

    def expand(self, possible_doses):
        for d in possible_doses:
            if d not in self.children:
                self.children[d] = MCTSNode(d, parent=self)
        return list(self.children.values())

    def is_leaf(self):
        return len(self.children) == 0

# ==========================================================
# Accurate MCTS Optimizer
# ==========================================================
class AccurateMCTSDoseOptimizer:
    def __init__(self, schedule="daily", n_iter=200, dose_space=None,
                 horizon=None, target=0.9,
                 bw_shift=None, comed_allowed=True,
                 use_clf=False, strict=True):
        self.schedule = schedule
        self.n_iter = n_iter
        if dose_space is None:
            if schedule == "daily":
                self.dose_space = list(np.arange(0.5, 25.5, 0.5))
                self.horizon = horizon or 24
            else:
                self.dose_space = list(np.arange(5, 55, 5))
                self.horizon = horizon or 168
        else:
            self.dose_space = dose_space
            self.horizon = horizon
        self.target = target
        self.bw_shift = bw_shift
        self.comed_allowed = comed_allowed
        self.use_clf = use_clf
        self.strict = strict

    def simulate_reward(self, dose):
        return simulate_population(
            dose,
            schedule=self.schedule,
            horizon=self.horizon,
            bw_shift=self.bw_shift,
            comed_allowed=self.comed_allowed,
            use_clf=self.use_clf,
            strict=self.strict
        )

    def run(self):
        root = MCTSNode(dose=None)
        for _ in range(self.n_iter):
            # Selection
            node = root
            while not node.is_leaf():
                node = node.best_child()

            # Expansion
            node.expand(self.dose_space)
            child = random.choice(list(node.children.values()))

            # Simulation
            reward = self.simulate_reward(child.dose)

            # Backpropagation
            while child is not None:
                child.visits += 1
                child.reward += reward
                child = child.parent

        best = max(root.children.values(), key=lambda c: c.reward / max(1, c.visits))
        return best.dose

# ==========================================================
# Tasks 1–5
# ==========================================================
def run_all_tasks():
    print("\n🌲 Accurate MCTS-based optimization\n")

    daily_opt = AccurateMCTSDoseOptimizer(schedule="daily", target=0.9).run()
    print(f"🎯 Task 1 (Daily, 90%): {daily_opt} mg")

    weekly_opt = AccurateMCTSDoseOptimizer(schedule="weekly", target=0.9).run()
    print(f"🎯 Task 2 (Weekly, 90%): {weekly_opt} mg")

    daily_shift = AccurateMCTSDoseOptimizer(schedule="daily", target=0.9, bw_shift="wider").run()
    weekly_shift = AccurateMCTSDoseOptimizer(schedule="weekly", target=0.9, bw_shift="wider").run()
    print(f"🎯 Task 3 (BW shift, Daily): {daily_shift} mg")
    print(f"🎯 Task 3 (BW shift, Weekly): {weekly_shift} mg")

    daily_nocomed = AccurateMCTSDoseOptimizer(schedule="daily", target=0.9, comed_allowed=False).run()
    weekly_nocomed = AccurateMCTSDoseOptimizer(schedule="weekly", target=0.9, comed_allowed=False).run()
    print(f"🎯 Task 4 (No COMED, Daily): {daily_nocomed} mg")
    print(f"🎯 Task 4 (No COMED, Weekly): {weekly_nocomed} mg")

    daily_75 = AccurateMCTSDoseOptimizer(schedule="daily", target=0.75).run()
    weekly_75 = AccurateMCTSDoseOptimizer(schedule="weekly", target=0.75).run()
    print(f"🎯 Task 5 (Daily, 75%): {daily_75} mg")
    print(f"🎯 Task 5 (Weekly, 75%): {weekly_75} mg")

    print("\n✅ All tasks completed with Accurate MCTS optimization!")

# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    run_all_tasks()
