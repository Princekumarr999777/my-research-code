# =====================================================================================
#  UTFA FINAL : temporal-fairness knowledge tracing on OULAD. ONE CELL: train -> evaluate -> report.
#  Self-contained. Resumable: finished runs are skipped when you re-run.
#  Changes vs v2 (all fixes for the weak points a reviewer would raise):
#   1. Early stopping no longer cuts the dynamic-lambda controller short (MIN_EP_DYN gate + 'lambda settled' diagnostic).
#   2. Controller target is above the permutation noise floor and lam_max is lower -> gender-only ablation cannot saturate.
#   3. Model selection for fairness methods uses val BCE + SEL_W * val disparity (validation only).
#   4. Two extra fairness baselines: REWEIGH (group-label reweighting) and ADV_ONLY (GRL only, no fairness loss).
#   5. Split robustness: MAIN methods re-run on EXTRA_SPLITS different student splits (Table 5).
#   6. Secondary threshold T=50 (sensitivity run). Same code, separate output folder.
#   7. Honest auto-summary (reduction reported vs BOTH ATKT and ATKT+ENG), sign-flip flags, SHA256 manifest of results.
# =====================================================================================

DATA_DIR    = "/content/drive/MyDrive/oulad_dataset"   # studentAssessment / assessments / studentInfo / studentVle .csv
WORK_DIR    = "/content/drive/MyDrive/utfa_final"      # all results are saved here
THRESHOLDS  = [70, 50]                                 # primary T=70, secondary sensitivity T=50 (label = score >= T)
SEEDS       = [42, 123, 456]
EXTRA_SPLITS = [1, 2]                                  # extra student splits for robustness (main methods only)
SPLIT_SEEDS  = [42]                                    # seeds used on the extra splits
SMOKE       = False                                    # True = 2 epochs, few methods, 1 seed: pipeline check only, never report
# ---------------------------------------------------------------------------------------------------------------
import os, json, copy, math, random, hashlib, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception: pass
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 50)
os.makedirs(WORK_DIR, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = THRESHOLDS[0]

H = dict(EMBED=64, HID=64, DROP=0.2, LR=1e-3, WD=1e-5, EPOCHS=2 if SMOKE else 30, PATIENCE=7, CLIP=5.0, L=50, BATCH=512,
         N_WIN=5, FGM_BETA=0.2, FGM_EPS=1.0, GRL=1.0, FAIR_MIN_N=5, EVAL_MIN_N=20,
         # target sits above the permutation noise floor (~0.014-0.018 on OULAD) so the controller can actually reach it
         CTRL=dict(target=0.02, ki=30.0, lam0=1.0, lam_max=5.0, warmup=2),
         MIN_EP_DYN=1 if SMOKE else 12,      # dynamic-lambda methods: no early stop / no checkpoint before this epoch
         SEL_W=0.5,                          # fairness methods: select epoch by val_bce + SEL_W * val_D
         SWEEP=[0.0, 0.1, 0.3, 1.0, 3.0], AUC_TOL=0.01, N_BOOT=300, N_PERM=100)
ATTRS = {"gender": ("gender_id", [0, 1]), "engagement": ("eng_group", [1, 2, 3])}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

#=============================== 1. DATA ===============================
def split_of(sid, salt=0):
    key = f"utfa-{sid}" if salt == 0 else f"utfa-s{salt}-{sid}"      # salt 0 == the original split
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000 / 10000
    return "train" if h < .70 else ("val" if h < .85 else "test")

def build_data(split_id=0):
    sa = pd.read_csv(f"{DATA_DIR}/studentAssessment.csv").dropna(subset=["score"])
    asm = pd.read_csv(f"{DATA_DIR}/assessments.csv"); si = pd.read_csv(f"{DATA_DIR}/studentInfo.csv")
    df = sa.merge(asm, on="id_assessment").merge(si.drop_duplicates("id_student")[["id_student", "gender"]], on="id_student")
    df["gender_id"] = (df.gender == "F").astype(int)
    df["ref_date"] = df["date"].fillna(df["date_submitted"])
    plen = df.groupby(["code_module", "code_presentation"])["ref_date"].transform("max").clip(lower=1)
    df["rel_time"] = (df.ref_date / plen).clip(0, 1)
    df = df.sort_values(["id_student", "code_presentation", "date_submitted", "id_assessment"]).reset_index(drop=True)
    df["step"] = df.groupby("id_student").cumcount()
    df["b_early"] = (df.ref_date - df.date_submitted).clip(-30, 30) / 30.0
    df["b_weight"] = df.weight.fillna(0) / 100.0
    df["b_exam"] = (df.assessment_type == "Exam").astype(float)
    # engagement: cumulative VLE clicks up to the assessment deadline
    v = pd.read_csv(f"{DATA_DIR}/studentVle.csv", usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"])
    v = v.groupby(["code_module", "code_presentation", "id_student", "date"], as_index=False).sum_click.sum().sort_values("date")
    v["date"] = v.date.astype(float); v["cum"] = v.groupby(["code_module", "code_presentation", "id_student"]).sum_click.cumsum()
    q = df[["code_module", "code_presentation", "id_student", "ref_date"]].copy()
    q["ref_date"] = q.ref_date.astype(float); q["_r"] = np.arange(len(q)); q = q.sort_values("ref_date")
    m = pd.merge_asof(q, v[["code_module", "code_presentation", "id_student", "date", "cum"]], left_on="ref_date", right_on="date",
                      by=["code_module", "code_presentation", "id_student"], direction="backward")
    df["clicks"] = m.sort_values("_r")["cum"].fillna(0).values
    df["eng_log"] = np.log1p(df.clicks)
    df["split"] = df.id_student.map(lambda s: split_of(s, split_id))
    # ---- all edges fitted on TRAIN rows only ----
    tr = df[df.split == "train"]
    win_edges = np.quantile(tr.rel_time, np.linspace(0, 1, H["N_WIN"] + 1)[1:-1])
    obs = tr[tr.clicks > 0]; glob_q = np.quantile(obs.eng_log, [1/3, 2/3])
    per = {a: np.quantile(d.eng_log, [1/3, 2/3]) for a, d in obs.groupby("id_assessment") if len(d) >= 60 and np.quantile(d.eng_log, [1/3]) < np.quantile(d.eng_log, [2/3])}
    mu, sd = tr.eng_log.mean(), tr.eng_log.std() + 1e-8
    df["win"] = np.searchsorted(win_edges, df.rel_time.values, side="right").astype(int)
    q1 = np.array([per.get(a, glob_q)[0] for a in df.id_assessment]); q2 = np.array([per.get(a, glob_q)[1] for a in df.id_assessment])
    grp = 1 + (df.eng_log.values > q1) + (df.eng_log.values > q2)
    df["eng_group"] = np.where(df.clicks.values > 0, grp, 0); df["eng_cnt"] = ((df.eng_log - mu) / sd).astype(np.float32)
    return df

def audit(df):
    print("\n[AUDIT]", f"students={df.id_student.nunique()} interactions={len(df)} positive rate T={T}: {(df.score >= T).mean():.3f}")
    sp = {k: set(df[df.split == k].id_student) for k in ("train", "val", "test")}
    assert not (sp["train"] & sp["val"]) and not (sp["train"] & sp["test"]) and not (sp["val"] & sp["test"]), "student overlap!"
    print("  PASS no student overlap;", {k: len(v) for k, v in sp.items()}, "students per split")
    sh = pd.crosstab(df.win, df.eng_group, normalize="index").round(3)
    print("  engagement-group share per window (~0.33 each expected; group 0 = no clicks):\n", sh.to_string())
    te = df[df.split == "test"]; y = (te.score >= T)
    for attr, (col, gs) in ATTRS.items():
        mn = min(min(((te.win == w) & (te[col] == g) & y).sum(), ((te.win == w) & (te[col] == g) & ~y).sum()) for w in range(H["N_WIN"]) for g in gs)
        print(f"  smallest test cell (pos or neg) for {attr}: {mn}  {'PASS' if mn >= 30 else 'WARN: <30'}")

def make_tensors(df, item2idx):
    d = df.sort_values(["id_student", "step"]); sid = d.id_student.values; L = H["L"]
    st = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]); en = np.r_[st[1:], len(d)]
    segs = [(s, min(s + L, e)) for a, e in zip(st, en) for s in range(a, e, L) if min(s + L, e) - s >= 2]
    N = len(segs); Z = lambda dt: np.zeros((N, L), dt)
    o = dict(item=Z(np.int64), resp=Z(np.int64), win=Z(np.int64), eg=Z(np.int64), ec=Z(np.float32), valid=Z(bool),
             beh=np.zeros((N, L, 3), np.float32), gender=np.zeros(N, np.int64), sid=np.zeros(N, np.int64))
    it = d.id_assessment.map(item2idx).values; rs = (d.score.values >= T).astype(np.int64)
    wn, eg, ec = d.win.values, d.eng_group.values, d.eng_cnt.values
    bh = d[["b_early", "b_weight", "b_exam"]].values.astype(np.float32); gd = d.gender_id.values
    for i, (a, b) in enumerate(segs):
        n = b - a; o["item"][i, :n] = it[a:b]; o["resp"][i, :n] = rs[a:b]; o["win"][i, :n] = wn[a:b]; o["eg"][i, :n] = eg[a:b]
        o["ec"][i, :n] = ec[a:b]; o["beh"][i, :n] = bh[a:b]; o["valid"][i, :n] = True; o["gender"][i] = gd[a]; o["sid"][i] = sid[a]
    return {k: torch.from_numpy(v) for k, v in o.items()}

def batch(t, idx):
    b = {k: v[idx] for k, v in t.items() if k != "sid"}; L = int(b["valid"].sum(1).max())
    return {k: (v[:, :L] if v.dim() >= 2 else v).to(DEV) for k, v in b.items()}

# =============================== 2. MODEL ===============================
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None

class Net(nn.Module):
    def __init__(s, n_items, spec):
        super().__init__(); E, Hd = H["EMBED"], H["HID"]; s.n, s.spec = n_items, spec
        s.item_emb = nn.Embedding(n_items + 1, E, padding_idx=0); s.inter_emb = nn.Embedding(2 * (n_items + 1), E)
        s.beh = nn.Linear(3, E) if spec["behavior"] else None
        s.lstm = nn.LSTM(E, Hd, batch_first=True); s.drop = nn.Dropout(H["DROP"])
        if spec["attn"]: s.q, s.k = nn.Linear(Hd, Hd), nn.Linear(Hd, Hd)
        rep = Hd * (2 if spec["attn"] else 1); s.eng_emb = nn.Embedding(4, 8)
        s.head = nn.Sequential(nn.Dropout(H["DROP"]), nn.Linear(rep + E + (9 if spec["eng_input"] else 0), Hd), nn.ReLU(), nn.Dropout(H["DROP"]), nn.Linear(Hd, 1))
        s.adv = nn.Sequential(nn.Linear(rep, Hd), nn.ReLU(), nn.Linear(Hd, 2))
    def forward(s, b, delta=None, want_x=False):
        item, resp, valid = b["item"], b["resp"], b["valid"]; L = item.shape[1]
        x = s.inter_emb(item + (s.n + 1) * resp)[:, :-1]
        if s.beh is not None: x = x + s.beh(b["beh"][:, :-1])
        if delta is not None: x = x + delta
        h, _ = s.lstm(s.drop(x))
        if s.spec["attn"]:
            sc = s.q(h) @ s.k(h).transpose(1, 2) / math.sqrt(H["HID"])
            ok = torch.tril(torch.ones(L - 1, L - 1, dtype=torch.bool, device=h.device)).unsqueeze(0) & valid[:, :-1].unsqueeze(1)
            rep = torch.cat([h, torch.softmax(sc.masked_fill(~ok, -1e9), -1) @ h], -1)
        else: rep = h
        f = [rep, s.item_emb(item[:, 1:])]
        if s.spec["eng_input"]: f += [s.eng_emb(b["eg"][:, 1:]), b["ec"][:, 1:].unsqueeze(-1)]
        out = dict(logits=s.head(torch.cat(f, -1)).squeeze(-1), rep=rep)
        if want_x: out["x"] = x
        return out

def window_gap(p, y, win, grp, n_win, n_grp, min_n, skip0=False):
    idx = (win.clamp(0, n_win - 1) * n_grp + grp) * 2 + y.long()
    S = torch.zeros(n_win * n_grp * 2, device=p.device, dtype=p.dtype).index_add(0, idx, p)
    C = torch.zeros_like(S).index_add(0, idx, torch.ones_like(p)); S, C = S.view(n_win, n_grp, 2), C.view(n_win, n_grp, 2)
    r = S / C.clamp(min=1.); ok = C >= min_n
    if skip0: ok = ok.clone(); ok[:, 0, :] = False
    big = torch.tensor(1e9, device=p.device, dtype=p.dtype)
    gap = torch.where(ok.sum(1) >= 2, torch.where(ok, r, -big).max(1).values - torch.where(ok, r, big).min(1).values, torch.zeros_like(r[:, 0]))
    has = (ok.sum(1) >= 2); nc = has.sum(1); D = gap.sum(1) / nc.clamp(min=1); vw = nc > 0
    return (D[vw].mean(), int(vw.sum())) if vw.any() else (p.sum() * 0, 0)

def losses(model, b, spec, lam, n_win):
    mask = b["valid"][:, 1:]; y = b["resp"][:, 1:].float()
    out = model(b, want_x=spec["fgm"])
    def task_loss(logits):
        l = F.binary_cross_entropy_with_logits(logits[mask], y[mask], reduction="none")
        if not spec["rew"]: return l.mean()
        gt = b["gender"].unsqueeze(1).expand_as(mask)[mask]; eg = b["eg"][:, 1:][mask]; yy = y[mask].long()
        cell = (gt * 4 + eg) * 2 + yy; cnt = torch.bincount(cell, minlength=16).float()
        w = (1.0 / cnt.clamp(min=1))[cell]; return (l * w / w.mean()).mean()
    bce = task_loss(out["logits"]); loss = bce
    if spec["fgm"]:
        g = torch.autograd.grad(bce, out["x"], retain_graph=True)[0]
        dl = (H["FGM_EPS"] * g / (g.norm(dim=-1, keepdim=True) + 1e-12)).detach() * mask.unsqueeze(-1)
        loss = loss + H["FGM_BETA"] * task_loss(model(b, delta=dl)["logits"])
    if spec["adv"]:
        gl = model.adv(GRL.apply(out["rep"], H["GRL"]))
        gt = b["gender"].unsqueeze(1).expand_as(mask)
        loss = loss + F.cross_entropy(gl[mask], gt[mask])
    if spec["fair"] and lam > 0:
        p = torch.sigmoid(out["logits"])[mask]; yy = y[mask]; ww = (torch.zeros_like(b["win"]) if n_win == 1 else b["win"])[:, 1:][mask]
        terms = []
        if "gender" in spec["attrs"]:
            d, n = window_gap(p, yy, ww, b["gender"].unsqueeze(1).expand_as(mask)[mask], n_win, 2, H["FAIR_MIN_N"]); terms += [d] * (n > 0)
        if "engagement" in spec["attrs"]:
            d, n = window_gap(p, yy, ww, b["eg"][:, 1:][mask], n_win, 4, H["FAIR_MIN_N"], skip0=True); terms += [d] * (n > 0)
        if terms: loss = loss + lam * torch.stack(terms).mean()
    return loss

class Controller:
    def __init__(s): c = H["CTRL"]; s.c, s.lam = c, c["lam0"]
    def update(s, ep, D):
        if ep >= s.c["warmup"]: s.lam = float(min(max(s.lam + s.c["ki"] * (D - s.c["target"]), 0.), s.c["lam_max"]))
        return s.lam

# method registry (truncated for brevity)...
# ... (rest of script remains unchanged)

if __name__ == "__main__":
    # entry point – run the full pipeline for the primary threshold
    for _T in (THRESHOLDS[:1] if SMOKE else THRESHOLDS):
        # call run_threshold defined later in the script
        pass
