#!/usr/bin/env python3
"""
Hybrid QAOA Optimization for PK/PD Dose Selection (Tasks 1–5)
- Population-based suppression cost
- Strict suppression criterion (all times < threshold)
- Classification probability combined with regression
"""

import numpy as np
import torch
import pandas as pd
import json, pickle, random
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
    """Load model, scalers, features from one training directory"""
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

# ==========================================================
# Ensemble prediction
# ==========================================================
def create_test_batch(dose_schedule, time, bw, comed, bundle, batch_size=1):
    """Generate feature vector consistent with training FE for one bundle"""
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
    """Return mean PD biomarker and mean suppression prob across ensemble"""
    preds_reg, preds_clf = [], []
    for bundle in bundles:
        batch = create_test_batch(dose_schedule, time, bw, comed, bundle)
        with torch.no_grad():
            out = bundle["model"](batch)
            # Regression
            reg = bundle["pd_target_scaler"].inverse_transform(
                out["pd"]["pred"].cpu().numpy().reshape(-1, 1)
            )[0, 0]
            preds_reg.append(reg)
            # Classification
            out_clf = bundle["model"]._forward_clf(batch)
            prob = F.softmax(out_clf["pd"]["pred"], dim=1)[0, 1].item()
            preds_clf.append(prob)
    return np.mean(preds_reg), np.mean(preds_clf)

# ==========================================================
# Population setup
# ==========================================================
df = pd.read_csv("data/EstData.csv")
subjects = df.groupby("ID")[["BW","COMED"]].first().reset_index()
print(f"👥 Population size: {len(subjects)}")

baseline_threshold = 3.3

# ==========================================================
# Simulation
# ==========================================================
def simulate_population(dose, schedule="daily", horizon=24, bw_shift=None, comed_allowed=True):
    """Strict suppression: biomarker < threshold for entire horizon"""
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

        suppressed = True
        for t in range(horizon+1):
            biom, prob = ensemble_predict(dose_schedule, start_time+t, bw, comed)
            if biom >= baseline_threshold and prob < 0.5:
                suppressed = False
                break
        suppress_flags.append(suppressed)
    return np.mean(suppress_flags)

# ==========================================================
# Bit decoding
# ==========================================================
def decode_bits(bits, step=5):
    bits = [int(round(b)) for b in bits]
    val = sum(b << i for i, b in enumerate(bits[::-1]))
    return val * step

# ==========================================================
# Cost function
# ==========================================================
def dose_cost(bits, schedule="weekly", target=0.9,
              horizon=None, bw_shift=None, comed_allowed=True,
              penalty_scale=100):
    step = 0.5 if schedule == "daily" else 5
    dose = decode_bits(bits, step=step)
    if dose == 0:
        return 9999
    frac = simulate_population(dose, schedule=schedule,
                               horizon=horizon,
                               bw_shift=bw_shift,
                               comed_allowed=comed_allowed)
    penalty = max(0, target - frac) * penalty_scale
    return dose + penalty

# ==========================================================
# Hybrid QAOA Optimizer
# ==========================================================
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.optimizers import COBYLA
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_aer.primitives import Sampler

def hybrid_qaoa_optimize(n_bits=4, schedule="weekly",
                         target=0.9, horizon=None,
                         bw_shift=None, comed_allowed=True,
                         maxiter=30, shots=256):

    backend = Aer.get_backend("qasm_simulator")
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

    def objective(params):
        qc = ansatz.assign_parameters(params)
        qc_meas = QuantumCircuit(ansatz.num_qubits, ansatz.num_qubits)
        qc_meas.compose(qc, inplace=True)
        qc_meas.measure(range(ansatz.num_qubits), range(ansatz.num_qubits))

        job = sampler.run(qc_meas, shots=shots)
        result = job.result()
        counts = result.quasi_dists[0]

        exp_cost = 0
        for bitstring, prob in counts.items():
            bitstring_str = format(bitstring, f"0{n_bits}b") if isinstance(bitstring, int) else bitstring
            bits = [int(b) for b in bitstring_str[::-1]]
            cost = dose_cost(bits, schedule=schedule,
                             target=target,
                             horizon=horizon,
                             bw_shift=bw_shift,
                             comed_allowed=comed_allowed)
            exp_cost += prob * cost
        return exp_cost

    x0 = np.random.rand(ansatz.num_parameters)
    res = optimizer.minimize(fun=objective, x0=x0)
    opt_params = res.x

    qc = ansatz.assign_parameters(opt_params)
    qc_meas = QuantumCircuit(ansatz.num_qubits, ansatz.num_qubits)
    qc_meas.compose(qc, inplace=True)
    qc_meas.measure(range(ansatz.num_qubits), range(ansatz.num_qubits))

    job = sampler.run(qc_meas, shots=shots)
    result = job.result()
    counts = result.quasi_dists[0]

    best_bits, best_dose, best_cost = None, None, 1e9
    for bitstring, prob in counts.items():
        bitstring_str = format(bitstring, f"0{n_bits}b") if isinstance(bitstring, int) else bitstring
        bits = [int(b) for b in bitstring_str[::-1]]
        dose = decode_bits(bits, step=(0.5 if schedule=="daily" else 5))
        cost = dose_cost(bits, schedule=schedule,
                         target=target,
                         horizon=horizon,
                         bw_shift=bw_shift,
                         comed_allowed=comed_allowed)
        if cost < best_cost:
            best_bits, best_dose, best_cost = bits, dose, cost

    return best_bits, best_dose, best_cost

# ==========================================================
# Tasks
# ==========================================================
def run_all_tasks_hybrid():
    print("\n⚛ Hybrid QAOA Optimization (Strict Suppression + Classifier)\n")

    bits, dose, cost = hybrid_qaoa_optimize(n_bits=6, schedule="daily", target=0.9, horizon=24)
    print(f"🎯 Task 1 (Daily, 90%): dose={dose} mg | bits={bits} | cost={cost:.2f}")

    bits, dose, cost = hybrid_qaoa_optimize(n_bits=4, schedule="weekly", target=0.9, horizon=168)
    print(f"🎯 Task 2 (Weekly, 90%): dose={dose} mg | bits={bits} | cost={cost:.2f}")

    bits, dose, cost = hybrid_qaoa_optimize(n_bits=6, schedule="daily", target=0.9,
                                            horizon=24, bw_shift="wider")
    print(f"🎯 Task 3 (Daily, BW shift): dose={dose} mg | bits={bits} | cost={cost:.2f}")

    bits, dose, cost = hybrid_qaoa_optimize(n_bits=4, schedule="weekly", target=0.9,
                                            horizon=168, bw_shift="wider")
    print(f"🎯 Task 3 (Weekly, BW shift): dose={dose} mg | bits={bits} | cost={cost:.2f}")

    bits, dose, cost = hybrid_qaoa_optimize(n_bits=6, schedule="daily", target=0.9,
                                            horizon=24, comed_allowed=False)
    print(f"🎯 Task 4 (Daily, No COMED): dose={dose} mg | bits={bits} | cost={cost:.2f}")

    bits, dose, cost = hybrid_qaoa_optimize(n_bits=4, schedule="weekly", target=0.9,
                                            horizon=168, comed_allowed=False)
    print(f"🎯 Task 4 (Weekly, No COMED): dose={dose} mg | bits={bits} | cost={cost:.2f}")

    bits, dose, cost = hybrid_qaoa_optimize(n_bits=6, schedule="daily", target=0.75, horizon=24)
    print(f"🎯 Task 5 (Daily, 75%): dose={dose} mg | bits={bits} | cost={cost:.2f}")

    bits, dose, cost = hybrid_qaoa_optimize(n_bits=4, schedule="weekly", target=0.75, horizon=168)
    print(f"🎯 Task 5 (Weekly, 75%): dose={dose} mg | bits={bits} | cost={cost:.2f}")

    print("\n✅ Hybrid QAOA optimization completed!")

# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    run_all_tasks_hybrid()
