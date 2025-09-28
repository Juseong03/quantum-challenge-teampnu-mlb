#!/usr/bin/env python3
"""
Accurate MCTS Optimization (Constraint-Aware, Final+Fallback)
- Ensemble + Regression/Classification Suppression
- Quantum-inspired MCTS selection (UCT 기반)
- Grid Search fallback 추가
- Tasks 1–5 included
- Dose constraints enforced:
    * daily  : multiples of 0.5 mg (0.5–15)
    * weekly : multiples of 5 mg (5–80)
"""

import os, json, math, random, pickle, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from tqdm import trange

from utils.factory import create_model
from utils.helpers import get_device
from config import Config
from data.loaders import features_from_dose_history


# ==========================================================
# Seed Setter
# ==========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[set_seed] Seed fixed to {seed}")


def quantize_dose(schedule: str, dose: float) -> float:
    """Ensure dose respects problem constraints."""
    if schedule == "daily":
        step, lo, hi = 0.5, 0.5, 10.0
    elif schedule == "weekly":
        step, lo, hi = 5.0, 5.0, 70.0
    else:
        raise ValueError("Unknown schedule")
    q = round(dose / step) * step
    return float(np.clip(q, lo, hi))


def allowed_doses(schedule: str) -> np.ndarray:
    if schedule == "daily":
        return np.arange(0.5, 10.0 + 1e-9, 0.5)
    elif schedule == "weekly":
        return np.arange(5.0, 70.0 + 1e-9, 5.0)
    else:
        raise ValueError("Unknown schedule")


# ==========================================================
# Run dir helper
# ==========================================================
def create_run_dir(base: Path, tag: str = "mcts"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run_dir] Saving outputs under: {run_dir}")
    return run_dir


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ==========================================================
# Model Loader
# ==========================================================
def load_model_bundle(model_dir: Path, device):
    with open(model_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    with open(model_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    with open(model_dir / "features.pkl", "rb") as f:
        feats = pickle.load(f)

    model = create_model(config, None, feats["pk"], feats["pd"])
    checkpoint = torch.load(model_dir / "model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return {
        "model": model,
        "pk_scaler": scalers["pk_scaler"],
        "pd_scaler": scalers["pd_scaler"],
        "pk_target_scaler": scalers["pk_target_scaler"],
        "pd_target_scaler": scalers["pd_target_scaler"],
        "pk_features": feats["pk"],
        "pd_features": feats["pd"],
    }


# ==========================================================
# Feature builder & prediction
# ==========================================================
def create_feature_dataframe(dose_schedule, times, subjects):
    obs_rows, dose_rows = [], []
    for sid, row in subjects.iterrows():
        bw, comed = row["BW"], row["COMED"]
        for t in times:
            obs_rows.append({
                "ID": sid, "TIME": t, "DV": 0.0, "DVID": 2,
                "BW": bw, "COMED": comed,
                "DOSE": dose_schedule[-1][1] if dose_schedule else 0.0
            })
        for (dt, amt) in dose_schedule:
            dose_rows.append({"ID": sid, "TIME": dt, "AMT": amt})
    return pd.DataFrame(obs_rows), pd.DataFrame(dose_rows)


def build_batch_inputs(obs, dose, bundle, device):
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
        "pk": {"x": X_tensor},
        "pd": {"x": X_tensor},
    }, fe_out["ID"].values


def ensemble_predict_batch(dose_schedule, times, subjects,
                           bundles, device, baseline_threshold,
                           option="pred", clf_threshold=0.5):
    all_preds, all_probs = [], []
    for bundle in bundles:
        batch, ids = build_batch_inputs(
            *create_feature_dataframe(dose_schedule, times, subjects),
            bundle, device
        )
        with torch.no_grad():
            out = bundle["model"](batch)
            reg_vals = bundle["pd_target_scaler"].inverse_transform(
                out["pd"]["pred"].cpu().numpy().reshape(-1, 1)
            ).flatten()
            all_preds.append(reg_vals)
            if "pred_clf" in out["pd"]:
                logits = out["pd"]["pred_clf"]
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.append(probs)
            else:
                all_probs.append((reg_vals < baseline_threshold).astype(float))
    return ids, np.mean(all_preds, axis=0), np.mean(all_probs, axis=0)


# ==========================================================
# Simulation
# ==========================================================
def simulate_population_batch(
    dose, subjects, bundles, device, baseline_threshold,
    schedule="daily", horizon=24,
    bw_shift=None, comed_allowed=True,
    option="pred", logic="or", clf_threshold=0.5
):
    dose = quantize_dose(schedule, dose)

    if schedule == "daily":
        dose_schedule = [(24 * i, dose) for i in range(21)]
        start_time = 21 * 24
    else:
        dose_schedule = [(168 * i, dose) for i in range(4)]
        start_time = 4 * 168

    times = [start_time + t for t in range(horizon + 1)]
    subjs = subjects.copy()
    if not comed_allowed:
        subjs = subjs[subjs["COMED"] == 0].reset_index(drop=True)
    if subjs.empty:
        return 0.0
    if bw_shift == "wider":
        subjs["BW"] = np.clip(np.interp(subjs["BW"], [50, 100], [70, 140]), 70, 140)

    ids, biom, probs = ensemble_predict_batch(
        dose_schedule, times, subjs,
        bundles, device, baseline_threshold,
        option=option, clf_threshold=clf_threshold
    )

    suppress_flags = []
    for sid in np.unique(ids):
        mask = ids == sid
        biom_vals, prob_vals = biom[mask], probs[mask]
        if option == "pred":
            flag = (np.min(biom_vals) < baseline_threshold)
        elif option == "pred_clf":
            flag = (np.mean(prob_vals) > clf_threshold)
        elif option == "both":
            flag_pred = (np.min(biom_vals) < baseline_threshold)
            flag_clf = (np.mean(prob_vals) > clf_threshold)
            flag = (flag_pred and flag_clf) if logic == "and" else (flag_pred or flag_clf)
        else:
            flag = False
        suppress_flags.append(flag)

    return float(np.mean(suppress_flags))


# ==========================================================
# MCTS
# ==========================================================
class MCTSNode:
    def __init__(self, dose, parent=None):
        self.dose = dose
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.reward = 0.0
        self.achieved = None

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, possible_doses):
        for d in possible_doses:
            if d not in self.children:
                self.children[d] = MCTSNode(d, parent=self)
        return list(self.children.values())

    def select_child(self, c_param=1.4):
        return max(
            self.children.values(),
            key=lambda child: (
                (child.reward / (child.visits + 1e-9)) +
                c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-9))
            )
        )


class AccurateMCTSDoseOptimizer:
    def __init__(self, subjects, bundles, device, baseline_threshold,
                 schedule="daily", n_iter=200,
                 target=0.9, bw_shift=None, comed_allowed=True,
                 option="pred", logic="or", clf_threshold=0.5,
                 c_param=1.4):
        self.subjects = subjects
        self.bundles = bundles
        self.device = device
        self.baseline_threshold = baseline_threshold
        self.schedule = schedule
        self.n_iter = n_iter
        self.target = target
        self.bw_shift = bw_shift
        self.comed_allowed = comed_allowed
        self.option = option
        self.logic = logic
        self.clf_threshold = clf_threshold
        self.c_param = c_param
        self.dose_space = allowed_doses(schedule)
        self.horizon = 24 if schedule == "daily" else 168

    def run(self):
        root = MCTSNode(dose=None)
        for _ in trange(self.n_iter, desc=f"MCTS-{self.schedule}", ncols=100):
            node = root
            while not node.is_leaf():
                node = node.select_child(c_param=self.c_param)

            children = node.expand(self.dose_space)
            if not children:
                continue
            child = random.choice(children)

            achieved = simulate_population_batch(
                child.dose, self.subjects, self.bundles, self.device, self.baseline_threshold,
                schedule=self.schedule, horizon=self.horizon,
                bw_shift=self.bw_shift, comed_allowed=self.comed_allowed,
                option=self.option, logic=self.logic, clf_threshold=self.clf_threshold
            )
            reward = -abs(achieved - self.target)
            child.achieved = achieved

            while child is not None:
                child.visits += 1
                child.reward += reward
                child = child.parent

        # --- 후보 정리 ---
        candidates = [(c.dose, c.achieved) for c in root.children.values()
                      if c.visits > 0 and c.achieved is not None]
        if not candidates:
            return None, None

        valid = [(d, s) for d, s in candidates if s >= self.target]
        if valid:
            return min(valid, key=lambda x: x[0])

        # --- Grid Search fallback ---
        best = None
        for d in self.dose_space:
            s = simulate_population_batch(
                d, self.subjects, self.bundles, self.device, self.baseline_threshold,
                schedule=self.schedule, horizon=self.horizon,
                bw_shift=self.bw_shift, comed_allowed=self.comed_allowed,
                option=self.option, logic=self.logic, clf_threshold=self.clf_threshold
            )
            if s >= self.target:
                best = (d, s)
                break
        if best:
            return best

        # 최종 fallback: 목표와 가장 가까운 것
        return min(candidates, key=lambda x: abs(x[1] - self.target))


# ==========================================================
# Tasks
# ==========================================================
def run_all_tasks_mcts(run_dir: Path, subjects, bundles, device, baseline_threshold,
                       option="both", logic="or", clf_threshold=0.5, c_param=1.4):
    results = {}
    tasks = {
        "task1_daily_90": dict(schedule="daily", target=0.9),
        "task2_weekly_90": dict(schedule="weekly", target=0.9),
        "task3_daily_bwshift_90": dict(schedule="daily", target=0.9, bw_shift="wider"),
        "task3_weekly_bwshift_90": dict(schedule="weekly", target=0.9, bw_shift="wider"),
        "task4_daily_nocomed_90": dict(schedule="daily", target=0.9, comed_allowed=False),
        "task4_weekly_nocomed_90": dict(schedule="weekly", target=0.9, comed_allowed=False),
        "task5_daily_75": dict(schedule="daily", target=0.75),
        "task5_weekly_75": dict(schedule="weekly", target=0.75),
    }

    for name, kwargs in tasks.items():
        opt = AccurateMCTSDoseOptimizer(subjects, bundles, device, baseline_threshold,
                                        option=option, logic=logic, clf_threshold=clf_threshold,
                                        c_param=c_param, **kwargs)
        dose, supp = opt.run()
        if dose is not None:
            dose = quantize_dose(kwargs["schedule"], dose)   # ✅ 안전하게 보정
            results[name] = {"dose": dose, "supp": supp}
            print(f"{name}: {dose} mg (supp={supp:.1%})")
        else:
            results[name] = {"dose": None, "supp": None}
            print(f"{name}: no valid dose")

    save_json({
        "results": results,
        "_meta": {
            "option": option,
            "logic": logic,
            "clf_threshold": clf_threshold,
            "c_param": c_param
        }
    }, run_dir / "results_summary.json")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pred", "pred_clf", "both"], default="both")
    parser.add_argument("--logic", type=str, choices=["and", "or"], default="or")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_root", type=str, default="results/mcts_runs")
    parser.add_argument("--c_param", type=float, default=1.4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    model_paths = [Path("results/exp6/cascade/resqnn_moe/s42")]
    bundles = [load_model_bundle(p, device) for p in model_paths]

    df = pd.read_csv("data/EstData.csv")
    subjects = df.groupby("ID")[["BW", "COMED"]].first().reset_index()
    baseline_threshold = 3.3
    clf_threshold = 0.5

    run_dir = create_run_dir(Path(args.save_root), tag="mcts")
    run_all_tasks_mcts(run_dir, subjects, bundles, device, baseline_threshold,
                       option=args.mode, logic=args.logic,
                       clf_threshold=clf_threshold, c_param=args.c_param)

    print("\n✅ All MCTS tasks completed!")
    print(f"📦 Summary saved to {run_dir / 'results_summary.json'}")
