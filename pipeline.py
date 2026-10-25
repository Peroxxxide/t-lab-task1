"""
KION: next-week watch prediction with a two-stage recommender.

Stage 1: candidate generation (recent popularity + item-to-item co-occurrence).
Stage 2: gradient boosting ranker on
    (a) hand-crafted features,
    (b) features generated automatically from a small grammar
        relation x time kernel x measure x normalisation,
    (c) an automatically selected subset of (b), selected on validation only.

Time split (weeks Mon-Sun):
    train target week 2021-08-02..08  (features from history < 2021-08-02)
    val   target week 2021-08-09..15  (features from history < 2021-08-09)
    test  target week 2021-08-16..22  (features from history < 2021-08-16)

Usage:  python pipeline.py --data <dir with interactions.csv, users.csv, items.csv>
"""
import argparse
import json
import os
import time
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
P = argparse.ArgumentParser()
P.add_argument("--data", default="data")
P.add_argument("--out", default="results")
P.add_argument("--n_users", type=int, default=200_000)   # random users, full history kept
P.add_argument("--n_pop", type=int, default=50)          # popularity candidates per user
P.add_argument("--n_i2i", type=int, default=50)          # co-occurrence candidates per user
P.add_argument("--n_neg", type=int, default=20)          # sampled negatives per user for training
P.add_argument("--max_eval_users", type=int, default=8_000)
P.add_argument("--seed", type=int, default=42)
P.add_argument("--perm_rows", type=int, default=40_000)   # validation rows for permutation importance
P.add_argument("--model", choices=["hgb", "lgbm"], default="hgb")  # hgb = sklearn HistGradientBoosting
ARGS = P.parse_args()

CUTS = {"train": pd.Timestamp("2021-08-02"), "val": pd.Timestamp("2021-08-09"),
        "test": pd.Timestamp("2021-08-16")}
WEEK = pd.Timedelta(days=7)
K = 10
GBDT = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=False, random_state=0)
os.makedirs(ARGS.out, exist_ok=True)
TIMES = {}
RESULTS = {"config": {**vars(ARGS), "cuts": {k: str(v.date()) for k, v in CUTS.items()},
                      "gbdt": GBDT, "k": K}}


class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a):
        TIMES[self.name] = TIMES.get(self.name, 0.0) + time.perf_counter() - self.t
        print(f"  [{self.name}] {time.perf_counter() - self.t:.1f} s", flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
print("loading data", flush=True)
inter = pd.read_csv(os.path.join(ARGS.data, "interactions.csv"), parse_dates=["last_watch_dt"])
users = pd.read_csv(os.path.join(ARGS.data, "users.csv"))
items = pd.read_csv(os.path.join(ARGS.data, "items.csv"))
inter = inter.rename(columns={"user_id": "user", "item_id": "item", "last_watch_dt": "date",
                              "total_dur": "dur", "watched_pct": "pct"})
inter["pct"] = inter["pct"].fillna(0.0)

# sample users, keep their full history (as required by the task statement)
rng = np.random.default_rng(ARGS.seed)
all_users = inter["user"].unique()
sample = rng.choice(all_users, size=min(ARGS.n_users, len(all_users)), replace=False)
inter = inter[inter["user"].isin(sample)].reset_index(drop=True)
print(f"  sampled {len(sample):,} users, {len(inter):,} interactions", flush=True)

# item metadata
items = items.rename(columns={"item_id": "item"})
items["ctype"] = (items["content_type"] == "series").astype(np.int8)
items["decade"] = (items["release_year"] // 10 * 10).fillna(-1).astype(int)
items["for_kids"] = items["for_kids"].fillna(0).astype(np.int8)
items["age_rating"] = items["age_rating"].fillna(-1)
items["genre_list"] = items["genres"].fillna("").map(
    lambda s: [g.strip() for g in s.split(",") if g.strip()])
items["n_genres"] = items["genre_list"].map(len)
item_genres = items[["item", "genre_list"]].explode("genre_list").dropna()
item_genres = item_genres.rename(columns={"genre_list": "genre"})
ITEM_META = items.set_index("item")[["ctype", "decade", "release_year", "for_kids",
                                      "age_rating", "n_genres"]]

# user metadata (categorical codes; unknown users -> -1)
users = users.rename(columns={"user_id": "user"})
for c in ["age", "income", "sex"]:
    users[c] = users[c].astype("category").cat.codes.astype(np.int16)
users["kids_flg"] = users["kids_flg"].astype(np.int16)
USER_META = users.set_index("user")[["age", "income", "sex", "kids_flg"]]

inter = inter.merge(items[["item", "ctype", "decade"]], on="item", how="left")
inter = inter.merge(users[["user", "age", "income", "sex", "kids_flg"]], on="user", how="left")
for c in ["ctype", "decade", "age", "income", "sex", "kids_flg"]:
    inter[c] = inter[c].fillna(-1).astype(np.int32)


# ---------------------------------------------------------------------------
# stage 1: candidates
# ---------------------------------------------------------------------------
def split_at(cut):
    hist = inter[inter["date"] < cut]
    target = inter[(inter["date"] >= cut) & (inter["date"] < cut + WEEK)]
    # leakage guard: no feature row is on or after the cut
    assert hist["date"].max() < cut
    return hist, target


def candidates(hist, target, cut):
    """Candidates for warm users active in the target week; seen items removed."""
    warm = np.intersect1d(target["user"].unique(), hist["user"].unique())
    seen = hist[["user", "item"]]

    # popularity over the last 7 days before the cut
    recent = hist[hist["date"] >= cut - WEEK]
    pop = recent["item"].value_counts()
    pop_items = pop.index.to_numpy()[: ARGS.n_pop + 200]       # extra to survive seen-filtering
    cp = pd.DataFrame({"user": np.repeat(warm, len(pop_items)),
                       "item": np.tile(pop_items, len(warm)),
                       "pop_rank": np.tile(np.arange(len(pop_items)), len(warm))})

    # item-to-item cosine co-occurrence from the history
    uid = {u: k for k, u in enumerate(hist["user"].unique())}
    iid = {i: k for k, i in enumerate(hist["item"].unique())}
    inv_i = np.array(list(iid.keys()))
    X = sp.csr_matrix((np.ones(len(hist), np.float32),
                       (hist["user"].map(uid).to_numpy(), hist["item"].map(iid).to_numpy())),
                      shape=(len(uid), len(iid)))
    C = (X.T @ X).tocsr()
    C.setdiag(0); C.eliminate_zeros()
    deg = np.asarray(X.sum(0)).ravel()
    inv = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1)))
    S = (inv @ C @ inv).tocsr()
    # user profile = last 10 items
    last = hist.sort_values("date").groupby("user").tail(10)
    last = last[last["user"].isin(warm)]
    wu = {u: k for k, u in enumerate(warm)}
    R = sp.csr_matrix((np.ones(len(last), np.float32),
                       (last["user"].map(wu).to_numpy(), last["item"].map(iid).to_numpy())),
                      shape=(len(warm), len(iid)))
    Sc = (R @ S).tocsr()
    rows, cols, vals = [], [], []
    for r in range(Sc.shape[0]):
        a, b = Sc.indptr[r], Sc.indptr[r + 1]
        if b == a:
            continue
        idx, v = Sc.indices[a:b], Sc.data[a:b]
        top = np.argsort(-v)[: ARGS.n_i2i + 30]
        rows.append(np.full(len(top), r)); cols.append(idx[top]); vals.append(v[top])
    ci = pd.DataFrame({"user": warm[np.concatenate(rows)], "item": inv_i[np.concatenate(cols)],
                       "i2i_score": np.concatenate(vals)})

    # remove seen items, keep top-n of each source, union
    def drop_seen(df):
        m = df.merge(seen.assign(_s=1), on=["user", "item"], how="left")
        return m[m["_s"].isna()].drop(columns="_s")

    cp = drop_seen(cp).sort_values(["user", "pop_rank"]).groupby("user").head(ARGS.n_pop)
    ci = drop_seen(ci).sort_values(["user", "i2i_score"], ascending=[True, False])
    ci = ci.groupby("user").head(ARGS.n_i2i)
    ci["i2i_rank"] = ci.groupby("user").cumcount()
    cand = cp.merge(ci, on=["user", "item"], how="outer")
    cand["pop_rank"] = cand["pop_rank"].fillna(999)
    cand["i2i_rank"] = cand["i2i_rank"].fillna(999)
    cand["i2i_score"] = cand["i2i_score"].fillna(0.0)

    tgt = target[target["user"].isin(warm)][["user", "item"]].drop_duplicates()
    cand = cand.merge(tgt.assign(label=1), on=["user", "item"], how="left")
    cand["label"] = cand["label"].fillna(0).astype(np.int8)
    stats = dict(warm_users=int(len(warm)), target_pairs=int(len(tgt)),
                 cand_per_user=float(len(cand) / len(warm)),
                 cand_recall=float(cand["label"].sum() / len(tgt)))
    return cand, tgt, stats


# ---------------------------------------------------------------------------
# stage 2a: hand-crafted features
# ---------------------------------------------------------------------------
def manual_features(cand, hist, cut):
    h = hist.assign(d=(cut - hist["date"]).dt.days)          # d >= 1 by construction
    f = cand[["user", "item", "pop_rank", "i2i_score", "i2i_rank"]].copy()

    # user aggregates
    ug = h.groupby("user")
    u = pd.DataFrame({
        "u_cnt_all": ug.size(),
        "u_cnt_7": h[h.d <= 7].groupby("user").size(),
        "u_cnt_30": h[h.d <= 30].groupby("user").size(),
        "u_mean_pct": ug["pct"].mean(),
        "u_log_dur_30": np.log1p(h[h.d <= 30].groupby("user")["dur"].sum()),
        "u_days_last": ug["d"].min(),
        "u_days_first": ug["d"].max(),
        "u_share_series": ug["ctype"].mean(),
    }).fillna({"u_cnt_7": 0, "u_cnt_30": 0, "u_log_dur_30": 0})
    f = f.merge(u, left_on="user", right_index=True, how="left")
    f = f.merge(USER_META, left_on="user", right_index=True, how="left")

    # item aggregates
    ig = h.groupby("item")
    it = pd.DataFrame({
        "i_cnt_7": h[h.d <= 7].groupby("item").size(),
        "i_cnt_prev7": h[(h.d > 7) & (h.d <= 14)].groupby("item").size(),
        "i_cnt_30": h[h.d <= 30].groupby("item").size(),
        "i_cnt_all": ig.size(),
        "i_mean_pct": ig["pct"].mean(),
        "i_log_mean_dur": np.log1p(ig["dur"].mean()),
        "i_days_first": ig["d"].max(),
    })
    it[["i_cnt_7", "i_cnt_prev7", "i_cnt_30"]] = it[["i_cnt_7", "i_cnt_prev7", "i_cnt_30"]].fillna(0)
    it["i_trend"] = (it["i_cnt_7"] + 1) / (it["i_cnt_prev7"] + 1)
    f = f.merge(it, left_on="item", right_index=True, how="left")
    f = f.merge(ITEM_META, left_on="item", right_index=True, how="left")

    # pair: genre affinity (indirect link user -> genre <- item)
    ug_g = h[["user", "item"]].merge(item_genres, on="item")
    share = ug_g.groupby(["user", "genre"]).size() / h.groupby("user").size()
    share = share.rename("g_share").reset_index()
    pg = f[["user", "item"]].merge(item_genres, on="item").merge(share, on=["user", "genre"], how="left")
    pg["g_share"] = pg["g_share"].fillna(0.0)
    agg = pg.groupby(["user", "item"])["g_share"].agg(["max", "mean"]).rename(
        columns={"max": "p_genre_max", "mean": "p_genre_mean"})
    f = f.merge(agg, left_on=["user", "item"], right_index=True, how="left")
    # pair: content-type match
    f["p_ctype_match"] = np.where(f["ctype"] == 1, f["u_share_series"], 1 - f["u_share_series"])
    # pair: item popularity among users of the same age band, last 30 days
    a = h[h.d <= 30].groupby(["item", "age"]).size().rename("ia")
    at = h[h.d <= 30].groupby("age").size().rename("at")
    f = f.merge(a, left_on=["item", "age"], right_index=True, how="left")
    f = f.merge(at, left_on="age", right_index=True, how="left")
    f["p_item_age_share"] = f["ia"].fillna(0) / f["at"].fillna(1)
    f = f.drop(columns=["ia", "at"])
    return f.drop(columns=["user", "item"]).astype(np.float32)


# ---------------------------------------------------------------------------
# stage 2b: automatic features from a grammar
#   relation  : which entity / entity pair the events are grouped by
#   kernel    : exponential decay with half-life h days, or a hard window of w days
#   measure   : weighted count, or weighted completion (sum of watched_pct/100)
#   norm      : raw, or share of the parent total in the same kernel
# ---------------------------------------------------------------------------
RELATIONS = {
    # name: (group keys on history rows, keys on candidate pairs, parent keys for share)
    "user": (["user"], ["user"], None),
    "item": (["item"], ["item"], None),
    "user_genre": (["user", "genre"], ["user", "genre"], ["user"]),
    "user_ctype": (["user", "ctype"], ["user", "ctype"], ["user"]),
    "user_decade": (["user", "decade"], ["user", "decade"], ["user"]),
    "item_age": (["item", "age"], ["item", "age"], ["age"]),
    "item_income": (["item", "income"], ["item", "income"], ["income"]),
    "item_sex": (["item", "sex"], ["item", "sex"], ["sex"]),
    "item_kids": (["item", "kids_flg"], ["item", "kids_flg"], ["kids_flg"]),
}
KERNELS = [("exp", 1), ("exp", 3), ("exp", 7), ("exp", 14), ("exp", 30), ("exp", 90),
           ("box", 7), ("box", 30), ("all", 0)]
MEASURES = ["cnt", "done"]


def kernel_weight(d, kind, h):
    if kind == "exp":
        return np.power(0.5, d / h)
    if kind == "box":
        return (d <= h).astype(np.float64)
    return np.ones_like(d, dtype=np.float64)


def auto_feature_names():
    names = []
    for rel, (_, _, parent) in RELATIONS.items():
        for kind, hl in KERNELS:
            for m in MEASURES:
                for norm in (["raw"] + (["share"] if parent else [])):
                    names.append(f"{rel}|{kind}{hl}|{m}|{norm}")
    return names


def auto_features(cand, hist, cut, wanted=None):
    """Compute grammar features for candidate pairs. `wanted`: optional subset of names."""
    names = auto_feature_names() if wanted is None else list(wanted)
    need = {}
    for n in names:
        rel, ker, m, norm = n.split("|")
        need.setdefault(rel, set()).add((ker, m, norm))
    h = hist.assign(d=(cut - hist["date"]).dt.days.astype(np.float64))
    hg = h.merge(item_genres, on="item")                     # one row per (event, genre)
    pairs = cand[["user", "item"]].copy()
    pairs = pairs.merge(ITEM_META[["ctype", "decade"]], left_on="item", right_index=True, how="left")
    pairs = pairs.merge(USER_META, left_on="user", right_index=True, how="left")
    for c in ["ctype", "decade", "age", "income", "sex", "kids_flg"]:
        pairs[c] = pairs[c].fillna(-1).astype(np.int32)
    out = {}
    for rel, specs in need.items():
        hkeys, ckeys, parent = RELATIONS[rel]
        src = hg if "genre" in hkeys else h
        cols = {}
        for ker, m, norm in specs:
            kind, hl = ker[:3], int(ker[3:])
            w = kernel_weight(src["d"].to_numpy(), kind, hl)
            if m == "done":
                w = w * src["pct"].to_numpy() / 100.0
            cols[(ker, m)] = w
        tab = pd.DataFrame({f"{k}|{m}": v for (k, m), v in cols.items()})
        tab[hkeys] = src[hkeys].to_numpy()
        agg = tab.groupby(hkeys).sum()
        par = None
        if parent and any(norm == "share" for _, _, norm in specs):
            if rel == "user_genre":     # parent total = all events of the user (not per-genre rows)
                ptab = pd.DataFrame({f"{k}|{m}": (kernel_weight(h["d"].to_numpy(), k[:3], int(k[3:]))
                                                  * (h["pct"].to_numpy() / 100.0 if m == "done" else 1.0))
                                     for (k, m) in cols})
                ptab["user"] = h["user"].to_numpy()
                par = ptab.groupby("user").sum()
            else:
                par = tab.groupby(parent).sum().drop(columns=[c for c in hkeys if c not in parent],
                                                     errors="ignore")
        if "genre" in ckeys:
            left = pairs[["user", "item"]].reset_index().merge(item_genres, on="item")
        else:
            left = pairs[["user", "item"] + [c for c in ckeys if c not in ("user", "item")]].reset_index()
        m_ = left.merge(agg, left_on=ckeys, right_index=True, how="left")
        if par is not None:
            m_ = m_.merge(par.add_suffix("|P"), left_on=parent, right_index=True, how="left")
        for ker, m, norm in specs:
            col = f"{ker}|{m}"
            v = m_[col].fillna(0.0)
            if norm == "share":
                v = v / m_[col + "|P"].replace(0, np.nan)
                v = v.fillna(0.0)
            if "genre" in ckeys:      # multi-valued: aggregate over the item's genres (max)
                v = v.groupby(m_["index"]).max()
                v = v.reindex(pairs.index).fillna(0.0)
            else:
                v = pd.Series(v.to_numpy(), index=m_["index"].to_numpy()).reindex(pairs.index).fillna(0.0)
            out[f"{rel}|{ker}|{m}|{norm}"] = v.to_numpy(np.float32)
    return pd.DataFrame(out, index=cand.index)[names]


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------
def rank_metrics(cand, score, tgt, k=K):
    """Recall@k, NDCG@k, MAP@k over all warm target users; relevant set = all target items."""
    df = cand[["user", "item"]].assign(s=score)
    df = df.sort_values(["user", "s"], ascending=[True, False])
    df["r"] = df.groupby("user").cumcount()
    top = df[df["r"] < k].merge(tgt.assign(rel=1), on=["user", "item"], how="left")
    top["rel"] = top["rel"].fillna(0)
    n_rel = tgt.groupby("user").size()
    users_ = n_rel.index
    top["disc"] = 1.0 / np.log2(top["r"] + 2)
    dcg = (top["rel"] * top["disc"]).groupby(top["user"]).sum().reindex(users_).fillna(0)
    idcg = n_rel.clip(upper=k).map(lambda n: (1.0 / np.log2(np.arange(n) + 2)).sum())
    hits = top["rel"].groupby(top["user"]).sum().reindex(users_).fillna(0)
    top["cum"] = top.groupby("user")["rel"].cumsum()
    top["prec"] = top["cum"] / (top["r"] + 1)
    ap = (top["prec"] * top["rel"]).groupby(top["user"]).sum().reindex(users_).fillna(0)
    ap = ap / n_rel.clip(upper=k)
    return dict(recall=float((hits / n_rel).mean()), ndcg=float((dcg / idcg).mean()),
                map=float(ap.mean()), users=int(len(users_)), _u=(dcg / idcg))


def sample_users(cand, tgt, n, seed):
    us = tgt["user"].unique()
    if len(us) > n:
        us = np.random.default_rng(seed).choice(us, n, replace=False)
    return cand[cand["user"].isin(us)].reset_index(drop=True), tgt[tgt["user"].isin(us)]


def train_rows(cand, n_neg, seed):
    """All positives + n_neg random negatives per user, only users with >=1 positive."""
    pos_users = cand.loc[cand["label"] == 1, "user"].unique()
    c = cand[cand["user"].isin(pos_users)]
    neg = c[c["label"] == 0].sample(frac=1.0, random_state=seed).groupby("user").head(n_neg)
    return pd.concat([c[c["label"] == 1], neg]).sort_index()


LGBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=50,
            reg_lambda=1.0, random_state=0, deterministic=True, force_row_wise=True, verbose=-1)


def fit(X, y):
    if ARGS.model == "lgbm":
        from lightgbm import LGBMClassifier       # same capacity settings as the sklearn model
        return LGBMClassifier(**LGBM).fit(np.asarray(X, dtype=np.float32), y)
    return HistGradientBoostingClassifier(**GBDT).fit(X, y)


def predict(model, X):
    if ARGS.model == "lgbm":
        return model.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
    return model.predict_proba(X)[:, 1]


def evaluate(model, X, cand, tgt):
    s = predict(model, X)
    m = rank_metrics(cand, s, tgt)
    m["auc"] = float(roc_auc_score(cand["label"], s))
    return m


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
SETS = {}
for name, cut in CUTS.items():
    print(f"\n== {name}: cut {cut.date()}", flush=True)
    hist, target = split_at(cut)
    with Timer(f"candidates_{name}"):
        cand, tgt, st = candidates(hist, target, cut)
    if name == "train":
        cand = train_rows(cand, ARGS.n_neg, ARGS.seed)
    else:
        cand, tgt = sample_users(cand, tgt, ARGS.max_eval_users, ARGS.seed)
        st["eval_users"] = int(tgt["user"].nunique())
    st["rows"] = int(len(cand))
    print("  ", st, flush=True)
    with Timer(f"manual_features_{name}"):
        Xm = manual_features(cand, hist, cut)
    with Timer(f"auto_features_{name}"):
        Xa = auto_features(cand, hist, cut)
    SETS[name] = dict(cand=cand.reset_index(drop=True), tgt=tgt, Xm=Xm.reset_index(drop=True),
                      Xa=Xa.reset_index(drop=True), hist_max=str(hist["date"].max().date()))
    RESULTS.setdefault("splits", {})[name] = st
    # leakage guard on labels: every positive is in the target week, never in the history
    assert not cand.merge(hist[["user", "item"]], on=["user", "item"]).shape[0]

ytr = SETS["train"]["cand"]["label"].to_numpy()
EXP = {}
PER_USER = {}


def run(name, cols_m=None, cols_a=None):
    def X(split):
        parts = []
        if cols_m is not None:
            parts.append(SETS[split]["Xm"][cols_m])
        if cols_a is not None:
            parts.append(SETS[split]["Xa"][cols_a])
        return pd.concat(parts, axis=1)
    with Timer(f"fit_{name}"):
        mdl = fit(X("train"), ytr)
    res = {s: evaluate(mdl, X(s), SETS[s]["cand"], SETS[s]["tgt"]) for s in ("val", "test")}
    for s in ("val", "test"):
        PER_USER[(name, s)] = res[s].pop("_u")
    res["n_features"] = int(X("train").shape[1])
    res["fit_s"] = TIMES[f"fit_{name}"]
    EXP[name] = res
    print(f"  {name:28s} nf={res['n_features']:4d}  val ndcg {res['val']['ndcg']:.4f} "
          f"test ndcg {res['test']['ndcg']:.4f}  recall {res['test']['recall']:.4f}", flush=True)
    return mdl, X


print("\n== non-learned baselines", flush=True)
for s in ("val", "test"):
    c = SETS[s]["cand"]
    EXP.setdefault("popularity", {})[s] = rank_metrics(c, -c["pop_rank"].to_numpy(), SETS[s]["tgt"])
    EXP.setdefault("i2i_then_pop", {})[s] = rank_metrics(
        c, c["i2i_score"].to_numpy() * 1e3 - c["pop_rank"].to_numpy() * 1e-3, SETS[s]["tgt"])
    for b in ("popularity", "i2i_then_pop"):
        PER_USER[(b, s)] = EXP[b][s].pop("_u")

print("\n== GBDT experiments", flush=True)
MAN = list(SETS["train"]["Xm"].columns)
AUTO = list(SETS["train"]["Xa"].columns)
# BASE: what every GBDT gets - stage-1 scores and static user/item attributes.
# The comparison is therefore between hand-written and generated *aggregates* only.
BASE = ["pop_rank", "i2i_score", "i2i_rank"] + list(USER_META.columns) + list(ITEM_META.columns)
MAN_AGG = [c for c in MAN if c not in BASE]
run("gbdt_base", cols_m=BASE)
run("gbdt_manual", cols_m=MAN)
mdl_all, X_all = run("gbdt_auto_all", cols_m=BASE, cols_a=AUTO)

# ---- automatic selection on validation only: permutation importance (AUC drop) ----
with Timer("selection"):
    Xv = pd.concat([SETS["val"]["Xm"][BASE], SETS["val"]["Xa"][AUTO]], axis=1); yv = SETS["val"]["cand"]["label"].to_numpy()
    # stratified subsample of validation rows: all positives + random negatives (40k rows)
    pos = np.flatnonzero(yv == 1); neg = np.flatnonzero(yv == 0)
    srng = np.random.default_rng(ARGS.seed)
    sub = np.concatenate([pos, srng.choice(neg, min(len(neg), max(ARGS.perm_rows - len(pos), 0)), replace=False)])
    Xs = Xv.iloc[sub].to_numpy(np.float32).copy(); ys = yv[sub]
    base = roc_auc_score(ys, predict(mdl_all, Xs))
    imp = {}
    prng = np.random.default_rng(ARGS.seed + 1)
    for j, c in enumerate(AUTO, start=len(BASE)):      # permute generated columns only
        keep = Xs[:, j].copy(); Xs[:, j] = prng.permutation(keep)
        imp[c] = base - roc_auc_score(ys, predict(mdl_all, Xs))
        Xs[:, j] = keep
    order = sorted(AUTO, key=lambda c: -imp[c])
RESULTS["perm_importance_val_auc"] = {c: float(imp[c]) for c in order}

sel_curve = {}
for k_ in (10, 20, 40, 80):
    run(f"gbdt_auto_top{k_}", cols_m=BASE, cols_a=order[:k_])
    sel_curve[k_] = EXP[f"gbdt_auto_top{k_}"]["val"]["ndcg"]
best_k = max(sel_curve, key=sel_curve.get)                    # chosen on validation
RESULTS["selection"] = dict(val_ndcg_by_k={str(k): v for k, v in sel_curve.items()}, best_k=best_k)
run("gbdt_manual_plus_auto_best", cols_m=MAN, cols_a=order[:best_k])

# ---- window study: keep relations fixed, vary only the kernel (user_genre share and item count) ----
print("\n== kernel study", flush=True)
for ker in [f"{k}{h}" for k, h in KERNELS]:
    cols = [f"user_genre|{ker}|cnt|share", f"item|{ker}|cnt|raw", f"user|{ker}|cnt|raw",
            f"item_age|{ker}|cnt|share", f"user_ctype|{ker}|cnt|share"]
    run(f"kernel_{ker}", cols_m=BASE, cols_a=cols)
multi = []
for ker in [f"{k}{h}" for k, h in KERNELS if k == "exp"]:
    multi += [f"user_genre|{ker}|cnt|share", f"item|{ker}|cnt|raw", f"user|{ker}|cnt|raw",
              f"item_age|{ker}|cnt|share", f"user_ctype|{ker}|cnt|share"]
run("kernel_all_exp", cols_m=BASE, cols_a=multi)

# ---- paired bootstrap over test users: NDCG@10 difference vs the manual baseline ----
boot = {}
brng = np.random.default_rng(ARGS.seed + 2)
ref = PER_USER[("gbdt_manual", "test")]
idx = brng.integers(0, len(ref), size=(2000, len(ref)))
for name in EXP:
    if name == "gbdt_manual" or (name, "test") not in PER_USER:
        continue
    d = (PER_USER[(name, "test")].reindex(ref.index) - ref).to_numpy()
    bs = d[idx].mean(1)
    boot[name] = dict(diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
RESULTS["bootstrap_test_ndcg_vs_manual"] = boot
for n_, b in boot.items():
    print(f"  {n_:28s} dNDCG {b['diff']:+.4f}  95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")

if ARGS.model == "lgbm":
    import lightgbm
    RESULTS["config"]["lgbm"] = LGBM
    RESULTS["config"]["lightgbm_version"] = lightgbm.__version__
RESULTS["experiments"] = EXP
RESULTS["times_s"] = TIMES
RESULTS["n_auto_features"] = len(AUTO)
RESULTS["manual_features"] = MAN
RESULTS["base_features"] = BASE
RESULTS["manual_aggregates"] = MAN_AGG
RESULTS["auto_top"] = order[:80]
with open(os.path.join(ARGS.out, "results.json"), "w") as fh:
    json.dump(RESULTS, fh, indent=1, ensure_ascii=False)
print("\nsaved", os.path.join(ARGS.out, "results.json"))
