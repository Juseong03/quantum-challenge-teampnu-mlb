#!/usr/bin/env python3
"""
Population Simulation & Optimal Dose Finder (Grid-based, Refined Final)
- Dose search restricted to allowed grid:
    * Daily: multiples of 0.5 mg (0.5–15 mg)
    * Weekly: multiples of 5 mg (5–80 mg)
- Prefer smaller doses when multiple candidates satisfy threshold
- Results saved with metadata (seed, model, options)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle, json
import random
import argparse
from datetime import datetime

from utils.helpers import get_device
from utils.factory import create_model
from config import Config


# ==========================================================
# Seed Setter
# ==========================================================
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
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
# Ensemble Loader
# ==========================================================
def load_model_bundle(model_dir: Path, device):
    print(f"[load_model_bundle] Loading model from {model_dir}")
    with open(model_dir / "config.json", "r") as f:
        config_dict = json.load(f)
    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    with open(model_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    pk_scaler = scalers["pk_scaler"]
    pd_scaler = scalers["pd_scaler"]
    pk_target_scaler = scalers["pk_target_scaler"]
    pd_target_scaler = scalers["pd_target_scaler"]

    with open(model_dir / "features.pkl", "rb") as f:
        feats = pickle.load(f)
    pk_features = feats["pk"]
    pd_features = feats["pd"]

    model = create_model(config, None, pk_features, pd_features)
    checkpoint = torch.load(model_dir / "model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(f"[load_model_bundle] Loaded model with {len(pk_features)} pk_features, {len(pd_features)} pd_features")
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
# Batch Ensemble Prediction
# ==========================================================
def ensemble_predict_batch(dose_schedule, times, subjects, bundle_list, device, baseline_threshold):
    """Run predictions for all subjects × times in one batch."""
    from data.loaders import features_from_dose_history

    if subjects.empty:
        return np.array([]), np.array([])

    obs_list, dose_list = [], []
    for sid, row in subjects.iterrows():
        bw, comed = row["BW"], row["COMED"]
        for t in times:
            obs_list.append({
                "ID": sid, "TIME": t, "DV": 0.0, "DVID": 2,
                "BW": bw, "COMED": comed,
                "DOSE": dose_schedule[-1][1] if dose_schedule else 0.0
            })
        dose_list += [{"ID": sid, "TIME": ts, "AMT": a} for ts, a in dose_schedule]

    obs = pd.DataFrame(obs_list)
    dose = pd.DataFrame(dose_list)

    fe_out = features_from_dose_history(
        obs, dose,
        add_pk_baseline=False, add_pd_delta=False,
        target="dv", allow_future_dose=False,
        time_windows=None, add_decay_features=True,
        half_lives=[24, 48, 72]
    )

    all_preds, all_probs = [], []
    for bundle in bundle_list:
        X = fe_out[bundle["pk_features"]].values
        X_scaled = bundle["pk_scaler"].transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        batch = {"pk": {"x": X_tensor}, "pd": {"x": X_tensor}}

        with torch.no_grad():
            out = bundle["model"](batch)
            biom = bundle["pd_target_scaler"].inverse_transform(
                out["pd"]["pred"].cpu().numpy().reshape(-1, 1)
            ).flatten()
            all_preds.append(biom)

            if "pred_clf" in out["pd"]:
                logits = out["pd"]["pred_clf"]
                prob = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.append(prob)
            else:
                all_probs.append((biom < baseline_threshold).astype(float))

    return np.mean(all_preds, axis=0), np.mean(all_probs, axis=0)


# ==========================================================
# Optimized Population Simulation
# ==========================================================
def simulate_population(dose, subjects, bundles, device,
                        baseline_threshold, schedule="daily", horizon=24,
                        bw_shift=None, comed_allowed=True,
                        option="pred", logic="or",
                        clf_threshold=0.5):

    if schedule == "daily":
        dose_schedule = [(24 * i, dose) for i in range(21)]
        start_time = 21 * 24
    elif schedule == "weekly":
        dose_schedule = [(168 * i, dose) for i in range(4)]
        start_time = 4 * 168
    else:
        raise ValueError("Unknown schedule")

    sub = subjects.copy()
    if not comed_allowed:
        sub = sub[sub["COMED"] == 0].reset_index(drop=True)
    if sub.empty:
        return 0.0
    if bw_shift == "wider":
        sub["BW"] = np.clip(np.interp(sub["BW"], [50, 100], [70, 140]), 70, 140)

    times = start_time + np.arange(horizon + 1)
    biom_vals, clf_vals = ensemble_predict_batch(dose_schedule, times, sub, bundles, device, baseline_threshold)

    if biom_vals.size == 0:
        return 0.0

    suppress_flags = []
    n_subjects = len(sub)
    for i in range(n_subjects):
        subj_biom = biom_vals[i*(horizon+1):(i+1)*(horizon+1)]
        subj_clf  = clf_vals[i*(horizon+1):(i+1)*(horizon+1)]

        if option == "pred":
            flag = (subj_biom.min() < baseline_threshold)
        elif option == "pred_clf":
            flag = (subj_clf.mean() > clf_threshold)
        elif option == "both":
            flag_pred = (subj_biom.min() < baseline_threshold)
            flag_clf  = (subj_clf.mean() > clf_threshold)
            flag = (flag_pred and flag_clf) if logic == "and" else (flag_pred or flag_clf)
        else:
            raise ValueError("Unknown option")

        suppress_flags.append(flag)

    return float(np.mean(suppress_flags))


# ==========================================================
# Optimal Dose Finder (Grid Search)
# ==========================================================
def find_optimal_dose(subjects, bundles, device, baseline_threshold,
                      schedule="daily", option="pred", threshold=0.9,
                      logic="or", clf_threshold=0.5, prefer_small=True):
    """Find minimal dose achieving suppression ≥ threshold on allowed grid."""

    horizon = 24 if schedule == "daily" else 168
    candidates = allowed_doses(schedule)

    print(f"[find_optimal_dose] schedule={schedule}, option={option}, threshold={threshold} "
          f"(candidates {candidates[0]}..{candidates[-1]})")

    best_dose, best_frac, best_gap = None, None, float("inf")

    for dose in candidates:
        frac = simulate_population(
            dose, subjects, bundles, device, baseline_threshold,
            schedule=schedule, horizon=horizon, option=option,
            logic=logic, clf_threshold=clf_threshold
        )
        print(f"    Trying {dose:.1f} mg → suppression={frac:.1%}")
        if frac >= threshold:
            gap = abs(frac - threshold)
            if gap < best_gap or (gap == best_gap and prefer_small and (best_dose is None or dose < best_dose)):
                best_dose, best_frac, best_gap = dose, frac, gap

    if best_dose is None:
        print("    ❌ No dose found satisfying threshold")
        return None, 0.0
    else:
        best_dose = quantize_dose(schedule, best_dose)
        print(f"    ✅ Optimal: {best_dose:.1f} mg (supp {best_frac:.1%}, thr={threshold:.0%})")
        return best_dose, float(best_frac)


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Simulation & Optimal Dose Finder (Grid-based, Refined)")
    parser.add_argument("--mode", type=str, choices=["pred", "pred_clf", "both", "all"],
                        default="both", help="Prediction mode")
    parser.add_argument("--logic", type=str, choices=["and", "or"], default="or",
                        help="Logic for both mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    device = get_device()
    model_paths = [Path("results/exp6/cascade/resqnn_moe/s42")]
    bundles = [load_model_bundle(p, device) for p in model_paths]

    df = pd.read_csv("data/EstData.csv")
    subjects = df.groupby("ID")[["BW", "COMED"]].first().reset_index()
    print(f"[main] Loaded {len(subjects)} subjects")

    baseline_threshold = 3.3
    clf_threshold = 0.5

    opt_list = ["pred", "pred_clf", "both"] if args.mode == "all" else [args.mode]

    # 결과 저장 준비
    results = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("results/grid_runs") / f"grid_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for opt in opt_list:
        print(f"\n========== {opt.upper()} ==========")
        res_opt = {}

        # Task 1
        print("[Task 1] Daily optimal dose (90%)")
        res_opt["task1_daily"] = find_optimal_dose(subjects, bundles, device, baseline_threshold,
                                                   schedule="daily", option=opt,
                                                   threshold=0.9, logic=args.logic,
                                                   clf_threshold=clf_threshold)

        print("[Task 2] Weekly optimal dose (90%)")
        res_opt["task2_weekly"] = find_optimal_dose(subjects, bundles, device, baseline_threshold,
                                                    schedule="weekly", option=opt,
                                                    threshold=0.9, logic=args.logic,
                                                    clf_threshold=clf_threshold)

        # Task 3
        print("[Task 3] BW shift (Daily 10mg, Weekly 20mg)")
        dose_d, dose_w = quantize_dose("daily", 10), quantize_dose("weekly", 20)
        frac_d = simulate_population(dose_d, subjects, bundles, device, baseline_threshold,
                                     schedule="daily", bw_shift="wider",
                                     option=opt, logic=args.logic, clf_threshold=clf_threshold)
        frac_w = simulate_population(dose_w, subjects, bundles, device, baseline_threshold,
                                     schedule="weekly", bw_shift="wider",
                                     option=opt, logic=args.logic, clf_threshold=clf_threshold)
        res_opt["task3"] = {"daily": {"dose": dose_d, "supp": frac_d},
                            "weekly": {"dose": dose_w, "supp": frac_w}}
        print(f"    Daily({dose_d}mg) {frac_d:.1%}, Weekly({dose_w}mg) {frac_w:.1%}")

        # Task 4
        print("[Task 4] No COMED (Daily 10mg, Weekly 20mg)")
        dose_d, dose_w = quantize_dose("daily", 10), quantize_dose("weekly", 20)
        frac_d = simulate_population(dose_d, subjects, bundles, device, baseline_threshold,
                                     schedule="daily", comed_allowed=False,
                                     option=opt, logic=args.logic, clf_threshold=clf_threshold)
        frac_w = simulate_population(dose_w, subjects, bundles, device, baseline_threshold,
                                     schedule="weekly", comed_allowed=False,
                                     option=opt, logic=args.logic, clf_threshold=clf_threshold)
        res_opt["task4"] = {"daily": {"dose": dose_d, "supp": frac_d},
                            "weekly": {"dose": dose_w, "supp": frac_w}}
        print(f"    Daily({dose_d}mg) {frac_d:.1%}, Weekly({dose_w}mg) {frac_w:.1%}")

        # Task 5
        print("[Task 5] Daily optimal dose (75%)")
        res_opt["task5_daily"] = find_optimal_dose(subjects, bundles, device, baseline_threshold,
                                                   schedule="daily", option=opt,
                                                   threshold=0.75, logic=args.logic,
                                                   clf_threshold=clf_threshold)

        print("[Task 5] Weekly optimal dose (75%)")
        res_opt["task5_weekly"] = find_optimal_dose(subjects, bundles, device, baseline_threshold,
                                                    schedule="weekly", option=opt,
                                                    threshold=0.75, logic=args.logic,
                                                    clf_threshold=clf_threshold)

        results[opt] = res_opt

    # 메타데이터와 함께 저장
    summary = {
        "results": results,
        "_meta": {
            "mode": args.mode,
            "logic": args.logic,
            "clf_threshold": clf_threshold,
            "seed": args.seed,
            "models": [str(p) for p in model_paths]
        }
    }
    with open(save_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n✅ All GRID tasks completed!")
    print(f"📦 Summary saved to {save_dir / 'results_summary.json'}")
