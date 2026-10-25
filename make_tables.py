"""Строит таблицы раздела 4 README из results/results.json (числа вручную не набираются)."""
import json
import sys
from collections import defaultdict

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/results.json", encoding="utf-8"))
E, B, T = R["experiments"], R["bootstrap_test_ndcg_vs_manual"], R["times_s"]
N_EVAL = f'{R["config"]["max_eval_users"]:,}'.replace(",", " ")


def row(name, label):
    e = E[name]
    v, t = e["val"], e["test"]
    nf = e.get("n_features", "-")
    fit = f"{e['fit_s']:.0f}" if "fit_s" in e else "-"
    b = B.get(name)
    d = "эталон" if name == "gbdt_manual" else (
        f"{b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]" if b else "-")
    auc = f"{t['auc']:.3f}" if "auc" in t else "-"
    return (f"| {label} | {nf} | {v['ndcg']:.4f} | {t['ndcg']:.4f} | {t['recall']:.4f} | "
            f"{t['map']:.4f} | {auc} | {d} | {fit} |")


H = ("| Модель | Признаков | Val NDCG@10 | Test NDCG@10 | Test Recall@10 | Test MAP@10 | Test AUC | "
     "Test: разница NDCG@10 с ручным набором [95% CI] | Обучение, с |\n|---|---|---|---|---|---|---|---|---|")
main = [("popularity", "Популярность за 7 дней (без обучения)"),
        ("i2i_then_pop", "Item-to-item, затем популярность (без обучения)"),
        ("gbdt_base", "GBDT: только base (скоры кандидатов + статические атрибуты)"),
        ("gbdt_manual", "GBDT: base + ручные агрегаты"), ("gbdt_auto_all", "GBDT: base + все сгенерированные"),
        ("gbdt_auto_top10", "GBDT: base + top-10 сгенерированных"),
        ("gbdt_auto_top20", "GBDT: base + top-20 сгенерированных"),
        ("gbdt_auto_top40", "GBDT: base + top-40 сгенерированных"),
        ("gbdt_auto_top80", "GBDT: base + top-80 сгенерированных"),
        ("gbdt_manual_plus_auto_best",
         f"GBDT: base + ручные + top-{R['selection']['best_k']} сгенерированных")]
print("### Основное сравнение\n")
print(H)
for n, l in main:
    print(row(n, l))

print("\n### Исследование окон (те же 5 признаков, меняется только временное ядро)\n")
print(H)
for n in E:
    if n.startswith("kernel_"):
        k = n[len("kernel_"):]
        if k == "all0":
            lab = "вся история"
        elif k == "all_exp":
            lab = "все затухания вместе (30 признаков)"
        elif k.startswith("exp"):
            lab = f"затухание, полупериод {k[3:]} дн."
        else:
            lab = f"жёсткое окно {k[3:]} дн."
        print(row(n, lab))

print("\n### Данные и кандидаты\n")
print("| Выборка | Тёплых пользователей, активных в целевой неделе | Целевых пар | Кандидатов на пользователя | "
      "Recall кандидатов | Строк в выборке |")
print("|---|---|---|---|---|---|")
for s, st in R["splits"].items():
    print(f"| {s} | {st['warm_users']:,} | {st['target_pairs']:,} | {st['cand_per_user']:.1f} | "
          f"{st['cand_recall']:.3f} | {st['rows']:,} |")
print(f"\nПервые четыре колонки - по всем тёплым активным пользователям недели. Строки: для train - положительные "
      f"пары-кандидаты и по {R['config']['n_neg']} случайных отрицательных на пользователя (пользователи без положительных кандидатов "
      f"не используются); для val и test - все кандидаты "
      f"{N_EVAL} случайно выбранных пользователей, на которых считаются метрики.")

print("\n### Top-15 сгенерированных признаков по permutation importance на валидации (падение AUC)\n")
print("| # | Признак (связь, ядро, мера, нормировка) | Падение AUC |\n|---|---|---|")
for i, (k, v) in enumerate(list(R["perm_importance_val_auc"].items())[:15], 1):
    print(f"| {i} | `{k}` | {v:.5f} |")

print("\n### Важность по ядрам и по связям (сумма положительных падений AUC)\n")
byk, byr = defaultdict(float), defaultdict(float)
for k, v in R["perm_importance_val_auc"].items():
    rel, ker, _, _ = k.split("|")
    byk[ker] += max(v, 0); byr[rel] += max(v, 0)
print("| Ядро | Сумма |\n|---|---|")
for k, v in sorted(byk.items(), key=lambda x: -x[1]):
    print(f"| {k} | {v:.4f} |")
print("\n| Связь | Сумма |\n|---|---|")
for k, v in sorted(byr.items(), key=lambda x: -x[1]):
    print(f"| {k} | {v:.4f} |")
print("\nОбозначения: `exp<h>` - затухание с полупериодом h дней, `box<w>` - жёсткое окно w дней, "
      "`all0` - вся история; `cnt` - число событий, `done` - досмотренность; `raw` - как есть, `share` - доля.")

print("\n### Кривая отбора на валидации\n")
print("| k | Val NDCG@10 |\n|---|---|")
for k, v in R["selection"]["val_ndcg_by_k"].items():
    print(f"| {k} | {v:.4f} |")

print("\n### Время вычислений (секунды, основной прогон)\n")
print("| Шаг | train | val | test |\n|---|---|---|---|")
names = {"candidates": "кандидаты", "manual_features": "ручные признаки", "auto_features": "сгенерированные признаки"}
for step in ("candidates", "manual_features", "auto_features"):
    print(f"| {names[step]} | " + " | ".join(f"{T[f'{step}_{s}']:.1f}" for s in ("train", "val", "test")) + " |")
print(f"\nОтбор по permutation importance: {T['selection']:.0f} с.")
