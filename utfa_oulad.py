# ╔══════════════════════════════════════════════════════════════════════╗
# ║  UTFA – Temporal-Fairness Knowledge Tracing (OULAD)                ║
# ║  Google Colab Ready – Copy this entire file into a Colab cell      ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── STEP 1: Mount Google Drive ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── STEP 2: Install dependencies (torch is pre-installed on Colab) ────
!pip install -q scikit-learn matplotlib

# ── STEP 3: Imports ─────────────────────────────────────────────────────
import os, json, random, hashlib, time, warnings
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — CHANGE THESE PATHS IF NEEDED
# ════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("/content/drive/MyDrive/oulad_dataset")   # ← your OULAD CSVs
WORK_DIR = Path("/content/drive/MyDrive/utfa_results")    # ← outputs go here

T           = 70              # score threshold → binary pass/fail label
SEEDS       = [42, 123, 456]  # 3 seeds for mean ± std
SMOKE       = False           # set True for quick test (2 epochs, 1 seed)

PROTECTED_ATTRS = ["gender", "age_band", "imd_band", "disability"]

print("✓ Configuration loaded")
print(f"  DATA_DIR = {DATA_DIR}")
print(f"  WORK_DIR = {WORK_DIR}")

# ════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════════
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mkdir(p):
    p.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_prob, threshold=0.5):
    """Full evaluation metrics suite."""
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    bce = float(-np.mean(
        y_true * np.log(np.clip(y_prob, 1e-12, 1-1e-12)) +
        (1 - y_true) * np.log(np.clip(1 - y_prob, 1e-12, 1-1e-12))
    ))

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "roc_auc":   auc,
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "bce_loss":  bce,
    }

# ════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════
def load_oulad(data_dir):
    """
    Load & merge the 4 core OULAD CSVs.

    Merge strategy:
      studentAssessment
        ─(id_assessment)─► assessments
        ─(id_student, code_module, code_presentation)─► studentInfo
        ─(id_student, code_module, code_presentation)─► VLE (aggregated)
    """
    files = {
        "sa":   data_dir / "studentAssessment.csv",
        "asmt": data_dir / "assessments.csv",
        "info": data_dir / "studentInfo.csv",
        "vle":  data_dir / "studentVle.csv",
    }

    # Check all files exist
    for name, f in files.items():
        if not f.is_file():
            raise FileNotFoundError(
                f"❌ Missing: {f}\n"
                f"Make sure DATA_DIR points to the folder with the OULAD CSVs.\n"
                f"Download from: https://analyse.kmi.open.ac.uk/open_dataset"
            )

    sa   = pd.read_csv(files["sa"])
    asmt = pd.read_csv(files["asmt"])
    info = pd.read_csv(files["info"])
    vle  = pd.read_csv(files["vle"])

    # Coerce types (some fields may contain '?' or blanks)
    sa["score"]          = pd.to_numeric(sa["score"], errors="coerce")
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    asmt["weight"]       = pd.to_numeric(asmt["weight"], errors="coerce")
    asmt["date"]         = pd.to_numeric(asmt["date"], errors="coerce")

    # 1) studentAssessment ⟕ assessments (many-to-one on id_assessment)
    df = sa.merge(asmt, on="id_assessment", how="left")

    # 2) ⟕ studentInfo (composite key: student + module + presentation)
    df = df.merge(info,
                  on=["id_student", "code_module", "code_presentation"],
                  how="left")

    # 3) Aggregate VLE engagement per student/module/presentation
    #    (a direct row-level join would cause a many-to-many explosion)
    vle_agg = (
        vle
        .groupby(["id_student", "code_module", "code_presentation"])
        .agg(
            total_clicks        = ("sum_click", "sum"),
            num_vle_interactions = ("sum_click", "count"),
            distinct_vle_dates  = ("date", "nunique"),
            mean_daily_clicks   = ("sum_click", "mean"),
        )
        .reset_index()
    )
    df = df.merge(vle_agg,
                  on=["id_student", "code_module", "code_presentation"],
                  how="left")

    # 4) Derived feature: how late/early the submission was
    df["submission_lateness"] = df["date_submitted"] - df["date"]

    # 5) Drop rows where the student never submitted (no score)
    df = df.dropna(subset=["score"])

    print(f"  ✓ Loaded {len(df):,} assessment records "
          f"({df['id_student'].nunique():,} students, "
          f"{df['code_module'].nunique()} modules)")
    return df


# ════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════
def featurize(df, keep_protected=False):
    """
    Create feature matrix X and binary label y.
    Optionally returns a DataFrame of protected attributes for fairness.
    """
    df = df.copy()
    df["label"] = (df["score"] >= T).astype(int)

    # Save protected attributes before dropping
    protected_df = None
    if keep_protected:
        available = [c for c in PROTECTED_ATTRS if c in df.columns]
        protected_df = df[available].copy().reset_index(drop=True)

    # Drop: identifiers, target, leaky columns
    #   final_result = course outcome (known only AFTER course ends = LEAKAGE)
    drop_cols = ["id_student", "id_assessment", "score", "final_result"]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.fillna(0)

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  ✓ Features: {X.shape[1]} columns  |  "
          f"Class balance: {y.mean():.1%} positive ({y.sum():,}/{len(y):,})")
    return X, y, protected_df


# ════════════════════════════════════════════════════════════════════════
#  STRATIFIED TRAIN/TEST SPLIT
# ════════════════════════════════════════════════════════════════════════
def stratified_split(*arrays, y, test_ratio=0.2, seed=0):
    """
    Split multiple arrays consistently, preserving class distribution.
    Returns list of (train, test) tuples.
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.array(train_idx); rng.shuffle(train_idx)
    test_idx  = np.array(test_idx);  rng.shuffle(test_idx)

    return [(arr[train_idx], arr[test_idx]) for arr in arrays]


# ════════════════════════════════════════════════════════════════════════
#  MODEL 1: LOGISTIC REGRESSION
# ════════════════════════════════════════════════════════════════════════
def train_logreg(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000,
                             random_state=0, solver="lbfgs")
    clf.fit(X_tr, y_train)
    prob = clf.predict_proba(X_te)[:, 1]
    return compute_metrics(y_test, prob), prob


# ════════════════════════════════════════════════════════════════════════
#  MODEL 2: FEED-FORWARD NEURAL NETWORK
# ════════════════════════════════════════════════════════════════════════
def train_ffnn(X_train, y_train, X_test, y_test,
               epochs=30, lr=0.001, hidden=64, batch_size=256):

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)

    class FFNet(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden//2, 1), nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFNet(X_tr.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n = len(X_tr)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            xb = torch.from_numpy(X_tr[idx]).to(device)
            yb = torch.from_numpy(y_train[idx,None].astype(np.float32)).to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = model(torch.from_numpy(X_te).to(device)).cpu().numpy().flatten()

    return compute_metrics(y_test, probs), probs


# ════════════════════════════════════════════════════════════════════════
#  MODEL 3: LSTM (TEMPORAL/SEQUENTIAL)
# ════════════════════════════════════════════════════════════════════════
class LSTMDataset(Dataset):
    """Sliding-window sequences per student, ordered by date_submitted."""

    def __init__(self, df, seq_len=5):
        self.seqs = []
        meta  = ["id_student", "date_submitted"]
        label = "label"
        feats = [c for c in df.columns if c not in meta + [label]]

        df = df.sort_values(["id_student", "date_submitted"])
        for _, grp in df.groupby("id_student"):
            if len(grp) < seq_len:
                continue
            X = grp[feats].values.astype(np.float32)
            y = grp[label].values.astype(np.int64)
            for i in range(len(X) - seq_len + 1):
                self.seqs.append((X[i:i+seq_len], int(y[i+seq_len-1])))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s, l = self.seqs[idx]
        return torch.from_numpy(s), torch.tensor(l, dtype=torch.float32)


def prepare_lstm_df(df):
    """Prepare DataFrame for LSTM: keep id_student + date_submitted for sequencing."""
    df = df.copy()
    df = df.dropna(subset=["date_submitted"])
    df["label"] = (df["score"] >= T).astype(int)
    df = df.drop(columns=["id_assessment", "score", "final_result"], errors="ignore")
    df = df.fillna(0)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0.0)
    return df


def train_lstm(df, epochs=20, batch_size=64, lr=0.001,
               hidden_dim=64, seq_len=5, seed=42):

    set_seeds(seed)
    lstm_df = prepare_lstm_df(df)
    dataset = LSTMDataset(lstm_df, seq_len=seq_len)

    if len(dataset) == 0:
        raise ValueError(
            f"No sequences of length {seq_len} could be formed. "
            f"Try seq_len=3 or seq_len=4."
        )

    n = len(dataset)
    split = int(0.8 * n)
    train_set, test_set = torch.utils.data.random_split(dataset, [split, n-split])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size)

    input_dim = dataset.seqs[0][0].shape[1]

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.fc   = nn.Linear(hidden_dim, 1)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return torch.sigmoid(self.fc(self.drop(h.squeeze(0)))).squeeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        for sb, lb in train_loader:
            sb, lb = sb.to(device), lb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sb), lb)
            loss.backward()
            optimizer.step()

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for sb, lb in test_loader:
            p = model(sb.to(device)).cpu().numpy()
            all_probs.extend(p.tolist())
            all_labels.extend(lb.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    return compute_metrics(y_true, y_prob), y_true, y_prob


# ════════════════════════════════════════════════════════════════════════
#  FAIRNESS ANALYSIS
# ════════════════════════════════════════════════════════════════════════
def compute_fairness(y_true, y_prob, protected_df):
    """Per-group metrics + Demographic Parity / Equalized Odds gaps."""
    results = {}
    for attr in protected_df.columns:
        groups = protected_df[attr].unique()
        group_metrics = {}
        pprs, tprs = [], []

        for g in sorted(groups, key=str):
            mask = (protected_df[attr] == g).values
            if mask.sum() < 10:
                continue
            m = compute_metrics(y_true[mask], y_prob[mask])
            m["n"] = int(mask.sum())
            group_metrics[str(g)] = m
            pprs.append(float((y_prob[mask] >= 0.5).mean()))
            tprs.append(m["recall"])

        dp = max(pprs) - min(pprs) if len(pprs) >= 2 else 0.0
        eo = max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0
        results[attr] = {
            "groups": group_metrics,
            "demographic_parity_diff": round(dp, 4),
            "equalized_odds_diff":     round(eo, 4),
        }
    return results


# ════════════════════════════════════════════════════════════════════════
#  RESULT SAVING / LOADING / AGGREGATION
# ════════════════════════════════════════════════════════════════════════
def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def aggregate_results(work_dir):
    """Aggregate per-seed JSONs → summary DataFrame with mean ± std."""
    records = []
    for f in sorted(work_dir.glob("*.json")):
        if f.name.startswith(("summary", "fairness")):
            continue
        data = json.loads(f.read_text())
        row = {"model": data["model"], "seed": data["seed"]}
        row.update(data["metrics"])
        records.append(row)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    metric_cols = [c for c in df.columns if c not in ("model", "seed")]
    rows = []
    for model, grp in df.groupby("model"):
        row = {"Model": model}
        for mc in metric_cols:
            vals = grp[mc].dropna()
            row[f"{mc}_mean"] = vals.mean()
            row[f"{mc}_std"]  = vals.std()
        rows.append(row)
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
#  LATEX TABLE GENERATION
# ════════════════════════════════════════════════════════════════════════
def make_latex_table(summary):
    """Generate a LaTeX results table you can paste into your paper."""
    if summary.empty:
        return ""

    names = {"logreg": "Logistic Regression",
             "shallow_nn": "Feed-Forward NN",
             "lstm": "LSTM"}
    keys  = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    heads = ["Accuracy", "AUC-ROC", "F1", "Precision", "Recall"]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Model performance on OULAD (mean $\pm$ std, "
            + str(len(SEEDS)) + r" seeds, threshold $T=" + str(T) + r"$)}",
        r"\label{tab:results}",
        r"\begin{tabular}{l" + "c"*len(keys) + "}",
        r"\toprule",
        "Model & " + " & ".join(heads) + r" \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        name = names.get(row["Model"], row["Model"])
        cells = []
        for k in keys:
            m = row.get(f"{k}_mean", float("nan"))
            s = row.get(f"{k}_std", float("nan"))
            cells.append(f"${m:.4f} \\pm {s:.4f}$" if not np.isnan(m) else "--")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex = "\n".join(lines)
    return tex


def make_fairness_latex(work_dir):
    """Generate a LaTeX fairness table."""
    ff = sorted(work_dir.glob("fairness_*.json"))
    if not ff:
        return ""

    agg = {}
    for f in ff:
        data = json.loads(f.read_text())
        for model, mdata in data.items():
            agg.setdefault(model, {})
            for attr, adata in mdata.items():
                agg[model].setdefault(f"{attr}_dp", []).append(adata["demographic_parity_diff"])
                agg[model].setdefault(f"{attr}_eo", []).append(adata["equalized_odds_diff"])

    names = {"logreg":"Logistic Regression","shallow_nn":"Feed-Forward NN","lstm":"LSTM"}
    ashort = {"gender":"Gender","age_band":"Age","imd_band":"IMD","disability":"Disability"}

    col_h = []
    for a in PROTECTED_ATTRS:
        s = ashort.get(a, a)
        col_h += [f"${s}$ $\\Delta$DP", f"${s}$ $\\Delta$EO"]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Fairness: Demographic Parity ($\Delta$DP) and Equalized Odds ($\Delta$EO) differences}",
        r"\label{tab:fairness}",
        r"\begin{tabular}{l" + "c"*len(col_h) + "}",
        r"\toprule",
        "Model & " + " & ".join(col_h) + r" \\",
        r"\midrule",
    ]
    for model in ["logreg", "shallow_nn", "lstm"]:
        if model not in agg:
            continue
        cells = []
        for a in PROTECTED_ATTRS:
            for sfx in ["_dp","_eo"]:
                vals = agg[model].get(f"{a}{sfx}", [])
                cells.append(f"${np.mean(vals):.4f}$" if vals else "--")
        lines.append(f"{names.get(model,model)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════════
def plot_results(summary, work_dir):
    if summary.empty:
        return
    names = {"logreg":"LogReg", "shallow_nn":"FF-NN", "lstm":"LSTM"}
    metrics = ["accuracy", "roc_auc", "f1"]
    labels  = ["Accuracy", "AUC-ROC", "F1 Score"]
    colors  = ["#4C72B0", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, met, lab in zip(axes, metrics, labels):
        models, means, stds = [], [], []
        for _, r in summary.iterrows():
            models.append(names.get(r["Model"], r["Model"]))
            means.append(r.get(f"{met}_mean", 0))
            stds.append(r.get(f"{met}_std", 0))
        bars = ax.bar(models, means, yerr=stds, color=colors[:len(models)],
                      capsize=5, edgecolor="black", linewidth=0.5)
        ax.set_title(lab, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        for b, m in zip(bars, means):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
                    f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(work_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved plot → {work_dir / 'model_comparison.png'}")


# ════════════════════════════════════════════════════════════════════════
#  ★ MAIN PIPELINE ★
# ════════════════════════════════════════════════════════════════════════
def run_seed(seed):
    """Full pipeline for one seed."""
    set_seeds(seed)

    # ── Load data ──────────────────────────────────────────────────────
    raw_df = load_oulad(DATA_DIR)
    X, y, prot_df = featurize(raw_df, keep_protected=True)

    # ── Stratified split ───────────────────────────────────────────────
    splits = stratified_split(X, y, prot_df.values, y=y, seed=seed)
    X_train, X_test = splits[0]
    y_train, y_test = splits[1]
    prot_test = pd.DataFrame(splits[2][1], columns=prot_df.columns)

    # ── Helper to save results ─────────────────────────────────────────
    fairness_all = {}

    def save(name, metrics):
        out = WORK_DIR / f"{name}_seed{seed}.json"
        if out.is_file():
            print(f"    [skip] {out.name} exists")
            return
        save_json(out, {
            "seed": seed, "model": name,
            "threshold": T,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
        })
        print(f"    [saved] {out.name}")

    # ── 1. Logistic Regression ─────────────────────────────────────────
    print(f"  ▸ Logistic Regression")
    lr_met, lr_prob = train_logreg(X_train, y_train, X_test, y_test)
    save("logreg", lr_met)
    fairness_all["logreg"] = compute_fairness(y_test, lr_prob, prot_test)
    print(f"    Acc={lr_met['accuracy']:.4f}  AUC={lr_met['roc_auc']:.4f}  "
          f"F1={lr_met['f1']:.4f}")

    # ── 2. Feed-Forward NN ─────────────────────────────────────────────
    print(f"  ▸ Feed-Forward NN")
    nn_ep = 2 if SMOKE else 30
    nn_met, nn_prob = train_ffnn(X_train, y_train, X_test, y_test, epochs=nn_ep)
    save("shallow_nn", nn_met)
    fairness_all["shallow_nn"] = compute_fairness(y_test, nn_prob, prot_test)
    print(f"    Acc={nn_met['accuracy']:.4f}  AUC={nn_met['roc_auc']:.4f}  "
          f"F1={nn_met['f1']:.4f}")

    # ── 3. LSTM ────────────────────────────────────────────────────────
    print(f"  ▸ LSTM")
    lstm_ep = 2 if SMOKE else 20
    try:
        lstm_met, _, _ = train_lstm(
            raw_df, epochs=lstm_ep, batch_size=64,
            hidden_dim=64, lr=0.001, seq_len=5, seed=seed
        )
        save("lstm", lstm_met)
        print(f"    Acc={lstm_met['accuracy']:.4f}  AUC={lstm_met['roc_auc']:.4f}  "
              f"F1={lstm_met['f1']:.4f}")
    except ValueError as e:
        print(f"    [warn] LSTM skipped: {e}")

    # ── Save fairness ──────────────────────────────────────────────────
    fp = WORK_DIR / f"fairness_seed{seed}.json"
    if not fp.is_file():
        save_json(fp, fairness_all)
        print(f"    [saved] {fp.name}")


# ──────────────────────────────────────────────────────────────────────
#  RUN EVERYTHING
# ──────────────────────────────────────────────────────────────────────
mkdir(WORK_DIR)
seeds = SEEDS[:1] if SMOKE else SEEDS

print("\n" + "="*60)
print("  UTFA – Temporal-Fairness Knowledge Tracing (OULAD)")
print("="*60)

for seed in seeds:
    print(f"\n{'─'*50}")
    print(f"  SEED {seed}")
    print(f"{'─'*50}")
    run_seed(seed)

# ── Aggregate & display results ────────────────────────────────────────
print(f"\n{'━'*50}")
print("  AGGREGATING RESULTS")
print(f"{'━'*50}\n")

summary = aggregate_results(WORK_DIR)

if not summary.empty:
    # ── Console table ──────────────────────────────────────────────────
    print("┌────────────────────────────────────────────────────────────────────┐")
    print("│                     RESULTS (mean ± std)                          │")
    print("├──────────────┬──────────┬──────────┬──────────┬──────────┬────────┤")
    print("│ Model        │ Accuracy │ AUC-ROC  │ F1       │ Prec     │ Recall │")
    print("├──────────────┼──────────┼──────────┼──────────┼──────────┼────────┤")
    nm = {"logreg":"LogReg", "shallow_nn":"FF-NN", "lstm":"LSTM"}
    for _, r in summary.iterrows():
        n = nm.get(r["Model"], r["Model"])
        print(f"│ {n:<12s} │ "
              f"{r.get('accuracy_mean',0):.4f}   │ "
              f"{r.get('roc_auc_mean',0):.4f}   │ "
              f"{r.get('f1_mean',0):.4f}   │ "
              f"{r.get('precision_mean',0):.4f}   │ "
              f"{r.get('recall_mean',0):.4f} │")
    print("└──────────────┴──────────┴──────────┴──────────┴──────────┴────────┘")

    # ── Save summary CSV ───────────────────────────────────────────────
    summary.to_csv(WORK_DIR / "summary.csv", index=False)
    print(f"\n✓ Summary CSV → {WORK_DIR / 'summary.csv'}")

    # ── LaTeX tables ───────────────────────────────────────────────────
    tex = make_latex_table(summary)
    (WORK_DIR / "results_table.tex").write_text(tex)
    print(f"✓ LaTeX table → {WORK_DIR / 'results_table.tex'}")

    ftex = make_fairness_latex(WORK_DIR)
    if ftex:
        (WORK_DIR / "fairness_table.tex").write_text(ftex)
        print(f"✓ Fairness table → {WORK_DIR / 'fairness_table.tex'}")

    # ── Plot ───────────────────────────────────────────────────────────
    plot_results(summary, WORK_DIR)

    # ── Print LaTeX for easy copy ──────────────────────────────────────
    print("\n" + "="*60)
    print("  LATEX TABLE (copy into your paper)")
    print("="*60)
    print(tex)

    if ftex:
        print("\n" + "="*60)
        print("  LATEX FAIRNESS TABLE")
        print("="*60)
        print(ftex)

print(f"\n✅ ALL DONE. Results saved to: {WORK_DIR}")
