#!/usr/bin/env python3
"""
Hybrid QAOA Optimization (Constraint-Aware, Final)
- Population-based suppression cost
- Regression/Classification suppression logic
- QAOA with classical cost evaluation + batch inference
- Tasks 1–5 included
- Dose constraints enforced:
    * daily  : multiples of 0.5 mg (0.5–15)
    * weekly : multiples of 5 mg (5–80)
"""

import numpy as np
import torch
import pandas as pd
import json, pickle, random
from pathlib import Path
from datetime import datetime
import argparse
from tqdm import tqdm

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
        step, lo, hi = 0.5, 0.5, 15.0   # ✅ 확장
    elif schedule == "weekly":
        step, lo, hi = 5.0, 5.0, 80.0   # ✅ 확장
    else:
        raise ValueError("Unknown schedule")
    q = round(dose / step) * step
    return float(np.clip(q, lo, hi))


def allowed_doses(schedule: str) -> np.ndarray:
    if schedule == "daily":
        return np.arange(0.5, 15.0 + 1e-9, 0.5)
    elif schedule == "weekly":
        return np.arange(5.0, 80.0 + 1e-9, 5.0)
    else:
        raise ValueError("Unknown schedule")


# ==========================================================
# Run directory helper
# ==========================================================
def create_run_dir(base: Path, tag: str = "qaoa"):
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run_dir] Saving outputs under: {run_dir}")
    return run_dir


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ==========================================================
# Model Loader
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
                           bundles, device, baseline_threshold=3.3, clf_threshold=0.5):
    if subjects.empty:
        return np.array([]), np.array([]), np.array([])
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
                probs = torch.sigmoid(out["pd"]["pred_clf"]).cpu().numpy().flatten()
            else:
                probs = (reg_vals < baseline_threshold).astype(float)
            all_probs.append(probs)
    return ids, np.mean(all_preds, axis=0), np.mean(all_probs, axis=0)


# ==========================================================
# Simulation
# ==========================================================
def simulate_population_batch(dose, subjects, bundles, device,
                              baseline_threshold, clf_threshold,
                              schedule="daily", horizon=24,
                              bw_shift=None, comed_allowed=True,
                              logic="or"):
    dose = quantize_dose(schedule, dose)

    if schedule == "daily":
        dose_schedule = [(24*i, dose) for i in range(21)]
        start_time = 21*24
    else:
        dose_schedule = [(168*i, dose) for i in range(4)]
        start_time = 4*168

    times = [start_time + t for t in range(horizon+1)]
    subjs = subjects.copy()
    if not comed_allowed:
        subjs = subjs[subjs["COMED"] == 0].reset_index(drop=True)
    if subjs.empty:
        return 0.0
    if bw_shift == "wider":
        subjs["BW"] = np.clip(np.interp(subjs["BW"], [50, 100], [70, 140]), 70, 140)

    ids, biom, probs = ensemble_predict_batch(
        dose_schedule, times, subjs,
        bundles, device, baseline_threshold, clf_threshold
    )
    if biom.size == 0:
        return 0.0

    suppress_flags = []
    for sid in np.unique(ids):
        mask = ids == sid
        biom_vals, prob_vals = biom[mask], probs[mask]
        flag_pred = (np.min(biom_vals) < baseline_threshold)
        flag_clf = (np.mean(prob_vals) > clf_threshold)
        flag = (flag_pred and flag_clf) if logic == "and" else (flag_pred or flag_clf)
        suppress_flags.append(flag)
    return float(np.mean(suppress_flags))


# ==========================================================
# Dose encoding & cost
# ==========================================================
def decode_bits(bits, schedule="daily"):
    bits = [int(round(b)) for b in bits]
    val = sum(b << i for i, b in enumerate(bits[::-1]))
    # schedule별 step 반영
    step = 0.5 if schedule == "daily" else 5.0
    dose = val * step
    return quantize_dose(schedule, dose)


def dose_cost(bits, subjects, bundles, device,
              baseline_threshold, clf_threshold,
              schedule="weekly", target=0.9, horizon=None,
              bw_shift=None, comed_allowed=True,
              alpha=1.0, beta=1000.0, logic="or"):
    dose = decode_bits(bits, schedule=schedule)
    frac = simulate_population_batch(dose, subjects, bundles, device,
                                     baseline_threshold, clf_threshold,
                                     schedule, horizon,
                                     bw_shift, comed_allowed, logic)
    penalty = max(0, target - frac) * beta
    cost = alpha * dose + penalty
    effective_frac = target if frac >= target else frac
    return cost, effective_frac, frac, dose



# ==========================================================
# QAOA Optimizer
# ==========================================================
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_aer.primitives import Sampler

def hybrid_qaoa_optimize(subjects, bundles, device,
                         baseline_threshold, clf_threshold,
                         n_bits=4, schedule="weekly", target=0.9, horizon=None,
                         bw_shift=None, comed_allowed=True,
                         logic="or", maxiter=30, shots=512, penalty_scale=1000):
    sampler = Sampler()
    coeffs = [1.0] * n_bits
    paulis = []
    for i in range(n_bits):
        z_str = ["I"] * n_bits
        z_str[i] = "Z"
        paulis.append(Pauli("".join(z_str)))
    H = SparsePauliOp.from_list([(str(p), c) for p, c in zip(paulis, coeffs)])
    ansatz = QAOAAnsatz(cost_operator=H, reps=2)
    optimizer = COBYLA(maxiter=maxiter)

    pbar = tqdm(total=maxiter, desc=f"QAOA-{schedule}-{target:.2f}", ncols=100)

    def objective(params):
        qc = ansatz.assign_parameters(params)
        qc_meas = QuantumCircuit(ansatz.num_qubits, ansatz.num_qubits)
        qc_meas.compose(qc, inplace=True)
        qc_meas.measure(range(ansatz.num_qubits), range(ansatz.num_qubits))
        result = sampler.run(qc_meas, shots=shots).result()
        counts = result.quasi_dists[0] if hasattr(result, "quasi_dists") else result.quasi_dist
        exp_cost = 0
        for bitstring, prob in counts.items():
            bits = [int(b) for b in format(bitstring, f"0{n_bits}b")[::-1]]
            cost, _, _ = dose_cost(bits, subjects, bundles, device,
                                   baseline_threshold, clf_threshold,
                                   schedule, target, horizon,
                                   bw_shift, comed_allowed,
                                   penalty_scale=penalty_scale, logic=logic)
            exp_cost += prob * cost
        pbar.update(1)
        return exp_cost

    x0 = np.random.rand(ansatz.num_parameters)
    res = optimizer.minimize(fun=objective, x0=x0)
    pbar.close()

    # --- 후처리 ---
    qc = ansatz.assign_parameters(res.x)
    qc_meas = QuantumCircuit(ansatz.num_qubits, ansatz.num_qubits)
    qc_meas.compose(qc, inplace=True)
    qc_meas.measure(range(ansatz.num_qubits), range(ansatz.num_qubits))
    result = sampler.run(qc_meas, shots=shots).result()
    counts = result.quasi_dists[0] if hasattr(result, "quasi_dists") else result.quasi_dist

    best_dose, best_cost, best_supp, raw_supp = None, 1e9, None, None
    for bitstring, prob in counts.items():
        bits = [int(b) for b in format(bitstring, f"0{n_bits}b")[::-1]]
        cost, supp, raw, dose = dose_cost(bits, subjects, bundles, device,
                                          baseline_threshold, clf_threshold,
                                          schedule, target, horizon,
                                          bw_shift, comed_allowed,
                                          alpha=1.0, beta=penalty_scale, logic=logic)
        if raw >= target and cost < best_cost:
            best_dose, best_cost, best_supp, raw_supp = dose, cost, supp, raw

    # --- fallback ---
    if best_dose is None:
        tqdm.write(f"⚠️ No valid dose ≥ {target:.0%}, fallback to best suppression")
        fallback = max(counts.items(), key=lambda kv: dose_cost(
            [int(b) for b in format(kv[0], f"0{n_bits}b")[::-1]],
            subjects, bundles, device, baseline_threshold, clf_threshold,
            schedule, target, horizon,
            bw_shift, comed_allowed, alpha=1.0, beta=penalty_scale, logic=logic
        )[2])  # pick by raw suppression
        bits = [int(b) for b in format(fallback[0], f"0{n_bits}b")[::-1]]
        _, best_supp, raw_supp, best_dose = dose_cost(bits, subjects, bundles, device,
                                                      baseline_threshold, clf_threshold,
                                                      schedule, target, horizon,
                                                      bw_shift, comed_allowed,
                                                      alpha=1.0, beta=penalty_scale, logic=logic)
    tqdm.write(f"✅ Best dose={best_dose:.2f} mg (supp={best_supp:.1%}, raw={raw_supp:.1%}, cost={best_cost:.2f})")

    return quantize_dose(schedule, best_dose), best_supp, raw_supp


# ==========================================================
# Tasks
# ==========================================================
def run_all_tasks_qaoa(run_dir: Path, subjects, bundles, device,
                       baseline_threshold, clf_threshold,
                       logic="or", seed=42, models=None):
    print("\n⚛ Hybrid QAOA Optimization (Tasks 1–5)\n")
    results = {}
    task_defs = {
        "task1_daily_90": dict(n_bits=7, schedule="daily", target=0.9, horizon=24),
        "task2_weekly_90": dict(n_bits=7, schedule="weekly", target=0.9, horizon=168),
        "task3_daily_bwshift_90": dict(n_bits=7, schedule="daily", target=0.9, horizon=24, bw_shift="wider"),
        "task3_weekly_bwshift_90": dict(n_bits=7, schedule="weekly", target=0.9, horizon=168, bw_shift="wider"),
        "task4_daily_nocomed_90": dict(n_bits=7, schedule="daily", target=0.9, horizon=24, comed_allowed=False),
        "task4_weekly_nocomed_90": dict(n_bits=7, schedule="weekly", target=0.9, horizon=168, comed_allowed=False),
        "task5_daily_75": dict(n_bits=7, schedule="daily", target=0.75, horizon=24),
        "task5_weekly_75": dict(n_bits=7, schedule="weekly", target=0.75, horizon=168),
    }
    for name, kwargs in task_defs.items():
        dose, supp, raw = hybrid_qaoa_optimize(subjects, bundles, device,
                                               baseline_threshold, clf_threshold,
                                               logic=logic, **kwargs)
        if dose is not None:
            results[name] = {"dose": quantize_dose(kwargs["schedule"], dose),
                             "supp": supp, "raw_supp": raw}
            print(f"{name}: {dose} mg (supp={supp:.1%}, raw={raw:.1%})")
        else:
            results[name] = {"dose": None, "supp": None, "raw_supp": None}
            print(f"{name}: no valid dose")

    summary = {"results": results, "_meta": {"seed": seed, "logic": logic,
                                             "clf_threshold": clf_threshold,
                                             "baseline_threshold": baseline_threshold,
                                             "models": models}}
    save_json(summary, run_dir / "results_summary.json")
    print("\n✅ All QAOA tasks completed!")
    print(f"📦 Summary saved to {run_dir / 'results_summary.json'}")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic", type=str, choices=["and", "or"], default="or")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_root", type=str, default="results/qaoa_runs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    model_paths = [Path("results/exp6/cascade/resqnn_moe/s42")]
    bundles = [load_model_bundle(p, device) for p in model_paths]

    df = pd.read_csv("data/EstData.csv")
    subjects = df.groupby("ID")[["BW", "COMED"]].first().reset_index()
    baseline_threshold, clf_threshold = 3.3, 0.5

    run_dir = create_run_dir(Path(args.save_root), tag="qaoa")
    run_all_tasks_qaoa(run_dir, subjects, bundles, device,
                       baseline_threshold, clf_threshold,
                       logic=args.logic, seed=args.seed,
                       models=[str(p) for p in model_paths])
