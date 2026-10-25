"""Build markdown tables for the README from results/results.json (no numbers are typed by hand)."""
import json
import sys
from collections import defaultdict

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/results.json"))
E, B, T = R["experiments"], R["bootstrap_test_ndcg_vs_manual"], R["times_s"]


def row(name, label):
    e = E[name]
    v, t = e["val"], e["test"]
    nf = e.get("n_features", "-")
    fit = f"{e['fit_s']:.0f}" if "fit_s" in e else "-"
    b = B.get(name)
    d = "ref." if name == "gbdt_manual" else (
        f"{b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]" if b else "-")
    auc = f"{t['auc']:.3f}" if "auc" in t else "-"
    return (f"| {label} | {nf} | {v['ndcg']:.4f} | {t['ndcg']:.4f} | {t['recall']:.4f} | "
            f"{t['map']:.4f} | {auc} | {d} | {fit} |")


H = ("| Model | Features | Val NDCG@10 | Test NDCG@10 | Test Recall@10 | Test MAP@10 | Test AUC | "
     "Test dNDCG vs manual [95% CI] | Fit, s |\n|---|---|---|---|---|---|---|---|---|")
main = [("popularity", "Popularity (last 7 days)"), ("i2i_then_pop", "Item-to-item, then popularity"),
        ("gbdt_base", "GBDT, base only (stage-1 scores + static attributes)"),
        ("gbdt_manual", "GBDT, base + hand-crafted aggregates"), ("gbdt_auto_all", "GBDT, base + all generated"),
        ("gbdt_auto_top10", "GBDT, base + generated top-10"), ("gbdt_auto_top20", "GBDT, base + generated top-20"),
        ("gbdt_auto_top40", "GBDT, base + generated top-40"), ("gbdt_auto_top80", "GBDT, base + generated top-80"),
        ("gbdt_manual_plus_auto_best", f"GBDT, base + hand-crafted + generated top-{R['selection']['best_k']}")]
print("### Main comparison\n")
print(H)
for n, l in main:
    print(row(n, l))

print("\n### Kernel study (same 5 relations, only the time kernel changes)\n")
print(H)
for n in E:
    if n.startswith("kernel_"):
        k = n[len("kernel_"):]
        lab = {"all0": "all history", "all_exp": "all decays together (30 features)"}.get(
            k, k.replace("exp", "exp. decay, half-life ").replace("box", "hard window ") + " d")
        print(row(n, lab))

print("\n### Data and candidates\n")
print("| Split | Warm users | Target pairs | Candidates per user | Candidate recall | Rows used |")
print("|---|---|---|---|---|---|")
for s, st in R["splits"].items():
    print(f"| {s} | {st['warm_users']:,} | {st['target_pairs']:,} | {st['cand_per_user']:.1f} | "
          f"{st['cand_recall']:.3f} | {st['rows']:,} |")

print("\n### Top-15 generated features by permutation importance on validation (AUC drop)\n")
print("| # | Feature (relation, kernel, measure, normalisation) | AUC drop |\n|---|---|---|")
for i, (k, v) in enumerate(list(R["perm_importance_val_auc"].items())[:15], 1):
    print(f"| {i} | `{k}` | {v:.5f} |")

print("\n### Importance aggregated by kernel and by relation (sum of positive AUC drops)\n")
byk, byr = defaultdict(float), defaultdict(float)
for k, v in R["perm_importance_val_auc"].items():
    rel, ker, _, _ = k.split("|")
    byk[ker] += max(v, 0); byr[rel] += max(v, 0)
print("| Kernel | Sum |\n|---|---|")
for k, v in sorted(byk.items(), key=lambda x: -x[1]):
    print(f"| {k} | {v:.4f} |")
print("\n| Relation | Sum |\n|---|---|")
for k, v in sorted(byr.items(), key=lambda x: -x[1]):
    print(f"| {k} | {v:.4f} |")

print("\n### Selection curve on validation\n")
print("| k | Val NDCG@10 |\n|---|---|")
for k, v in R["selection"]["val_ndcg_by_k"].items():
    print(f"| {k} | {v:.4f} |")

print("\n### Compute (seconds, this machine)\n")
print("| Step | train | val | test |\n|---|---|---|---|")
for step in ("candidates", "manual_features", "auto_features"):
    print(f"| {step} | " + " | ".join(f"{T[f'{step}_{s}']:.1f}" for s in ("train", "val", "test")) + " |")
print(f"\nPermutation-importance selection: {T['selection']:.0f} s.")
