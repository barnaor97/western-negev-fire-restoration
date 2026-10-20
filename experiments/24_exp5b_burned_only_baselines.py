"""
24_exp5b_burned_only_baselines.py — Experiment 5b: burned-only controlled regression comparison.

Why. In Experiment 5 (pipeline/06_experiments.ipynb, cells 26-27) each Random Forest was fitted to ALL training
pixels, including the zero targets of pixels without a labelled fire month, and scored on ALL test pixels, whereas
the neural regression heads (L-TAE, BiGRU, TempCNN) receive regression loss only from burned pixels
(model_defs.MaskedMultiTaskLoss: burned_pixel = y.sum(over time) > 0). Table 14 therefore did not compare like
with like. This experiment keeps everything from Experiment 5 identical except ONE change - the Random Forest
regressors are trained on burned training pixels only - and evaluates every model on the same burned test pixels.
Experiment 5 itself is not modified.

Identical to Experiment 5: splits.json train/test indices; targets_aux.npz; feature representation
X_flat = concat(X.mean(axis=1), X.std(axis=1)) of the authoritative X_scaled.npy (104 features);
PCA(n_components=0.95, random_state=42) fitted on X_flat[train] (all training pixels, as in Experiment 5);
RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42), scikit-learn defaults otherwise.
Changed: regressor training rows = train ∩ burned.  Evaluation rows = test ∩ burned.

Neural models are NOT retrained: L-TAE from models/main/predictions_test.npz; BiGRU and TempCNN from their saved
checkpoints (06b Experiments 11-12), run in eval mode on the test sequences of the same X_scaled.npy.

Stages:
  extract  stream X_scaled.npy once (local file or `rclone cat` remote), verify its md5, write X_flat and the
           test-pixel sequences to --cache (outside the repository; ~2.3 GB)
  rf_one   fit/evaluate ONE forest per process (--rf-model, --rf-dim, --rf-pop burned|all) and save it at once
  rf_collect  merge the saved forests into rf_results.json and rf_burned_only_predictions.npz
  neural   checkpoint inference for BiGRU/TempCNN and the L-TAE predictions
Run: --stage extract --x-source bgu:thesisv4/data/processed/X_scaled.npy --cache C; --stage neural --cache C;
     for m in rf_raw rf_pca; for d in severity persistence recovery_gap: --stage rf_one --cache C --rf-model $m --rf-dim $d;
     --stage rf_one --cache C --rf-model rf_raw --rf-dim severity --rf-pop all  (Experiment 5 reproduction check);
     --stage rf_collect
Outputs: models/experiments/exp5b_burned_only_baselines/
"""
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "models", "experiments", "exp5b_burned_only_baselines")
CKPT = os.path.join(ROOT, "models", "experiments")   # published layout; was notebooks/figures_notebooks/thesisv4_results_20260813/models/experiments
SEED = 42
DIMS = ["severity", "persistence", "recovery_gap"]
EXPECTED_MD5 = "b34c14fb3d56ea69dc52b3861f4289f2"      # md5 of the authoritative X_scaled.npy (Drive, 2026-08-05)


def log(*a):
    print("[%s]" % datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def load_common():
    splits = json.load(open(os.path.join(P, "splits.json")))
    tr, va, te = (np.array(splits[k]) for k in ("train", "val", "test"))
    y = np.load(os.path.join(P, "y_burned.npy"), mmap_mode="r")
    burned = np.asarray(y).sum(axis=1) > 0                      # exactly the MaskedMultiTaskLoss definition
    targets = dict(np.load(os.path.join(P, "targets_aux.npz")))
    return tr, va, te, burned, targets


# ------------------------------------------------------------------ stage: extract
def stage_extract(source, cache):
    os.makedirs(cache, exist_ok=True)
    tr, va, te, burned, targets = load_common()
    if os.path.exists(source):
        proc = None; stream = open(source, "rb")
    else:
        proc = subprocess.Popen(["rclone", "cat", source], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 24)
        stream = proc.stdout
    md5 = hashlib.md5()
    def read_exact(n):
        buf = bytearray();
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk: raise IOError("stream ended early")
            buf += chunk
        md5.update(buf); return bytes(buf)
    # parse the .npy header
    magic = read_exact(8); assert magic[:6] == b"\x93NUMPY", "not an npy file"
    major = magic[6]
    hlen = int.from_bytes(read_exact(2 if major == 1 else 4), "little")
    header = eval(read_exact(hlen).decode("latin1"))
    shape, dtype, fortran = header["shape"], np.dtype(header["descr"]), header["fortran_order"]
    assert not fortran and dtype == np.float32 and len(shape) == 3, header
    N, T, C = shape; row = T * C * 4
    log("X_scaled.npy header:", shape, dtype)
    test_set = np.zeros(N, bool); test_set[te] = True
    test_pos = -np.ones(N, np.int64); test_pos[te] = np.arange(len(te))
    Xflat = np.empty((N, 2 * C), np.float32)
    Xtest = np.lib.format.open_memmap(os.path.join(cache, "X_test_sequences.npy"), mode="w+", dtype=np.float32, shape=(len(te), T, C))
    step = 4000; t0 = time.time()
    for a in range(0, N, step):
        b = min(a + step, N)
        blk = np.frombuffer(read_exact((b - a) * row), dtype=np.float32).reshape(b - a, T, C)
        Xflat[a:b] = np.concatenate([blk.mean(axis=1), blk.std(axis=1)], axis=1)   # Experiment 5, cell 26
        sel = np.where(test_set[a:b])[0]
        if len(sel): Xtest[test_pos[a + sel]] = blk[sel]
        if (a // step) % 25 == 0: log("pixels %d/%d  (%.0f MB/s)" % (b, N, b * row / 1e6 / max(time.time() - t0, 1e-6)))
    rest = stream.read()
    if rest: md5.update(rest)
    if proc: proc.wait()
    digest = md5.hexdigest()
    log("md5 of streamed X_scaled.npy:", digest, "MATCH" if digest == EXPECTED_MD5 else "MISMATCH")
    assert digest == EXPECTED_MD5, "X_scaled.npy md5 mismatch - not the authoritative array"
    Xtest.flush(); del Xtest
    np.save(os.path.join(cache, "X_flat.npy"), Xflat)
    json.dump(dict(source=source, md5=digest, shape=list(shape), n_test=int(len(te))), open(os.path.join(cache, "extract_info.json"), "w"), indent=1)


# ------------------------------------------------------------------ stage: rf
def rf_inputs(cache, name):
    """Feature matrix for one model; the PCA is deterministic (random_state=42, fitted on all training pixels as in Experiment 5)."""
    from sklearn.decomposition import PCA
    tr, va, te, burned, targets = load_common()
    Xflat = np.load(os.path.join(cache, "X_flat.npy"))
    pca = PCA(n_components=0.95, random_state=SEED).fit(Xflat[tr])
    info = dict(pca_components=int(pca.n_components_), pca_explained=float(pca.explained_variance_ratio_.sum()))
    X = Xflat if name == "rf_raw" else pca.transform(Xflat).astype(np.float32)
    return X, tr, te, burned, targets, info


def stage_rf_one(cache, name, dim, pop):
    """Fit ONE forest in its own process and save its metrics and predictions immediately.
    pop='burned': train ∩ burned -> test ∩ burned (the controlled change).
    pop='all'   : all train -> all test (Experiment 5 setup; used only as a reproduction check)."""
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import pearsonr, spearmanr
    import sklearn
    parts = os.path.join(OUT, "rf_parts"); os.makedirs(parts, exist_ok=True)
    tag = "%s_%s_%s" % (name, dim, pop)
    if os.path.exists(os.path.join(parts, tag + ".json")): log(tag, "already done"); return
    X, tr, te, burned, targets, info = rf_inputs(cache, name)
    rows_tr, rows_te = (tr[burned[tr]], te[burned[te]]) if pop == "burned" else (tr, te)
    log("%s: fit on %d rows, evaluate on %d rows" % (tag, len(rows_tr), len(rows_te)))
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=SEED)
    rf.fit(X[rows_tr], targets[dim][rows_tr])
    p = rf.predict(X[rows_te]).astype(np.float32); y = targets[dim][rows_te]
    r = dict(model=name, target=dim, population=pop, n_train=int(len(rows_tr)), n_test=int(len(rows_te)),
             pearson=float(pearsonr(p, y)[0]), spearman=float(spearmanr(p, y)[0]),
             mae=float(np.abs(p - y).mean()), rmse=float(np.sqrt(((p - y) ** 2).mean())),
             fit_seconds=round(time.time() - t0, 1), sklearn=sklearn.__version__, **info)
    np.save(os.path.join(parts, tag + "_pred.npy"), p)
    json.dump(r, open(os.path.join(parts, tag + ".json"), "w"), indent=1)
    log("%s r=%.6f (%.0fs)" % (tag, r["pearson"], r["fit_seconds"]))


def stage_rf_collect():
    tr, va, te, burned, targets = load_common()
    parts = os.path.join(OUT, "rf_parts"); te_b = te[burned[te]]
    res = dict(rf_raw={}, rf_pca={}); preds = {}
    for name in res:
        for d in DIMS:
            r = json.load(open(os.path.join(parts, "%s_%s_burned.json" % (name, d))))
            res[name][d] = r; preds["%s_%s" % (name, d)] = np.load(os.path.join(parts, "%s_%s_burned_pred.npy" % (name, d)))
    check = None
    cp = os.path.join(parts, "rf_raw_severity_all.json")
    if os.path.exists(cp):
        c = json.load(open(cp))
        check = dict(model="rf_raw", target="severity", population="all train / all test (Experiment 5 setup)",
                     reproduced_pearson=c["pearson"], published_pearson=0.8628197115230493,
                     abs_difference=abs(c["pearson"] - 0.8628197115230493), fit_seconds=c["fit_seconds"])
    np.savez_compressed(os.path.join(OUT, "rf_burned_only_predictions.npz"), test_indices_burned=te_b, **preds,
                        **{"target_" + d: targets[d][te_b] for d in DIMS})
    json.dump(dict(results=res, reproduction_check=check, n_train_all=int(len(tr)), n_train_burned=int(burned[tr].sum()),
                   n_test_all=int(len(te)), n_test_burned=int(len(te_b)),
                   pca_components=res["rf_pca"]["severity"]["pca_components"], pca_explained=res["rf_pca"]["severity"]["pca_explained"]),
              open(os.path.join(OUT, "rf_results.json"), "w"), indent=1)
    log("collected", {n: {d: round(res[n][d]["pearson"], 4) for d in DIMS} for n in res}, "| check:", check)


# ------------------------------------------------------------------ stage: neural
def stage_neural(cache):
    import torch, torch.nn as nn
    from scipy.stats import pearsonr, spearmanr
    os.makedirs(OUT, exist_ok=True)
    tr, va, te, burned, targets = load_common()
    mask = np.load(os.path.join(P, "X_scaled_mask.npy"), mmap_mode="r")
    Xte = np.load(os.path.join(cache, "X_test_sequences.npy"), mmap_mode="r")
    Mte = np.asarray(mask[te]).astype(bool)
    bt = burned[te]

    class GRUModel(nn.Module):                                           # verbatim from 06b, cell 13
        def __init__(self, n_features, hidden_dim=128, n_layers=1, dropout=0.1):
            super().__init__()
            self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0, bidirectional=True)
            d_out = hidden_dim * 2; self.dropout = nn.Dropout(dropout)
            self.burn_head = nn.Linear(d_out, 1); self.severity_head = nn.Linear(d_out, 1)
            self.persistence_head = nn.Linear(d_out, 1); self.recovery_gap_head = nn.Linear(d_out, 1)
        def forward(self, x, mask=None):
            out, _ = self.gru(x); out = self.dropout(out)
            m = mask.unsqueeze(-1).float(); seq_mean = (out * m).sum(dim=1) / (m.sum(dim=1) + 1e-6)
            return {"severity": torch.sigmoid(self.severity_head(seq_mean)).squeeze(-1),
                    "persistence": torch.sigmoid(self.persistence_head(seq_mean)).squeeze(-1),
                    "recovery_gap": torch.sigmoid(self.recovery_gap_head(seq_mean)).squeeze(-1)}

    class TempCNNBlock(nn.Module):                                       # verbatim from 06b, cell 15
        def __init__(self, i, o, k=5, dropout=0.1):
            super().__init__(); self.conv = nn.Conv1d(i, o, kernel_size=k, padding=k // 2); self.bn = nn.BatchNorm1d(o)
            self.relu = nn.ReLU(inplace=True); self.dropout = nn.Dropout(dropout)
        def forward(self, x): return self.dropout(self.relu(self.bn(self.conv(x))))

    class TempCNNModel(nn.Module):
        def __init__(self, n_features, hidden_dim=128, dropout=0.1, kernel_size=5):
            super().__init__()
            self.block1 = TempCNNBlock(n_features, hidden_dim, kernel_size, dropout)
            self.block2 = TempCNNBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            self.block3 = TempCNNBlock(hidden_dim, hidden_dim, kernel_size, dropout)
            self.burn_head = nn.Conv1d(hidden_dim, 1, kernel_size=1)
            self.severity_head = nn.Linear(hidden_dim, 1); self.persistence_head = nn.Linear(hidden_dim, 1)
            self.recovery_gap_head = nn.Linear(hidden_dim, 1)
        def forward(self, x, mask=None):
            h = self.block3(self.block2(self.block1(x.transpose(1, 2))))
            m = mask.unsqueeze(1).float(); pooled = (h * m).sum(dim=2) / (m.sum(dim=2) + 1e-6)
            return {"severity": torch.sigmoid(self.severity_head(pooled)).squeeze(-1),
                    "persistence": torch.sigmoid(self.persistence_head(pooled)).squeeze(-1),
                    "recovery_gap": torch.sigmoid(self.recovery_gap_head(pooled)).squeeze(-1)}

    def infer(model):
        model.eval(); outs = {d: [] for d in DIMS}
        with torch.no_grad():
            for a in range(0, len(te), 1024):
                o = model(torch.from_numpy(np.ascontiguousarray(Xte[a:a + 1024])), mask=torch.from_numpy(Mte[a:a + 1024]))
                for d in DIMS: outs[d].append(o[d].numpy())
        return {d: np.concatenate(v) for d, v in outs.items()}

    def score(pred, name, saved=None):
        r = {"all_test": {}, "burned_test": {}}
        for d in DIMS:
            y = targets[d][te]
            r["all_test"][d] = float(pearsonr(pred[d], y)[0])
            r["burned_test"][d] = dict(pearson=float(pearsonr(pred[d][bt], y[bt])[0]), spearman=float(spearmanr(pred[d][bt], y[bt])[0]))
        if saved: r["saved_all_test"] = {d: saved[d]["pearson"] for d in DIMS}
        log(name, "all:", {d: round(r["all_test"][d], 4) for d in DIMS}, "| burned:", {d: round(r["burned_test"][d]["pearson"], 4) for d in DIMS})
        return r

    res = {}
    lt = np.load(os.path.join(ROOT, "models", "main", "predictions_test.npz"))
    assert np.array_equal(lt["indices"], te)
    res["L-TAE"] = score({d: lt[d] for d in DIMS}, "L-TAE")
    for name, folder, ctor in (("BiGRU", "exp11_gru_baseline", lambda: GRUModel(52, 128, 1, 0.1)),
                               ("TempCNN", "exp12_tempcnn_baseline", lambda: TempCNNModel(52, 128, 0.1, 5))):
        m = ctor(); sd = torch.load(os.path.join(CKPT, folder, "model.pt"), map_location="cpu")
        m.load_state_dict(sd, strict=True)
        saved = json.load(open(os.path.join(CKPT, folder, "results.json")))["correlations"]
        res[name] = score(infer(m), name, saved)
        res[name]["checkpoint_sha256"] = sha256(os.path.join(CKPT, folder, "model.pt"))
    json.dump(dict(results=res, n_test_all=int(len(te)), n_test_burned=int(bt.sum())), open(os.path.join(OUT, "neural_results.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "rf_one", "rf_collect", "neural"], required=True)
    ap.add_argument("--x-source", default=os.path.join(P, "X_scaled.npy"))
    ap.add_argument("--cache")
    ap.add_argument("--rf-model", choices=["rf_raw", "rf_pca"])
    ap.add_argument("--rf-dim", choices=DIMS)
    ap.add_argument("--rf-pop", choices=["burned", "all"], default="burned")
    a = ap.parse_args()
    if a.stage == "extract": stage_extract(a.x_source, a.cache)
    if a.stage == "rf_one": stage_rf_one(a.cache, a.rf_model, a.rf_dim, a.rf_pop)
    if a.stage == "rf_collect": stage_rf_collect()
    if a.stage == "neural": stage_neural(a.cache)
