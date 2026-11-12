"""Эксперимент «старый vs новый датасет» (Assignment #2, вариант B).

Гипотеза: расширение исходного датасета синтетическими данными (трудные негативы,
казахский язык, новые типы мошенничества) улучшает обобщение модели на реальных,
ранее не виданных постах.

Контролируемый эксперимент — меняется ТОЛЬКО обучающий датасет:
  * модель одна и та же: `app.model.train.build_pipeline()`
    (TF-IDF word + char_wb + handcrafted-сигналы -> LogisticRegression);
  * нормализация текста одна и та же (`normalize_obfuscated`);
  * оценка на одном и том же отложенном наборе `data/eval_holdout.jsonl`
    (106 реалистичных постов, не входит ни в одну версию датасета).

Версии датасета — снимки `data/dataset.jsonl` из истории git (`data/versions/`):
v00 = исходный датасет (520 строк), v09 = итоговый расширенный (1953 строки).

Результат: `app/model/artifacts/compare.json` (для вкладки «Старый vs новый»)
+ две модели для живого сравнения: `clf_original.joblib`, `clf_enriched.joblib`.

Запуск: python -m scripts.compare_datasets
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit

try:  # кириллица в Windows-консоли
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import CATEGORIES
from app.model.normalize import normalize_obfuscated
from app.model.train import build_pipeline

ROOT = Path(__file__).resolve().parent.parent
VERSIONS_DIR = ROOT / "data" / "versions"
HOLDOUT_PATH = ROOT / "data" / "eval_holdout.jsonl"
ARTIFACTS = ROOT / "app" / "model" / "artifacts"
COMPARE_PATH = ARTIFACTS / "compare.json"
OLD_MODEL_PATH = ARTIFACTS / "clf_original.joblib"
NEW_MODEL_PATH = ARTIFACTS / "clf_enriched.joblib"

# Что добавлялось в датасет на каждом шаге (по сообщениям коммитов).
VERSION_NOTES = {
    "v00": "Исходный датасет: ядровые шаблоны, 130 примеров на класс",
    "v01": "Пограничные случаи и трудные негативы (новости, финграмотность, спорт)",
    "v02": "Реалистичные промо-формулировки из каналов",
    "v03": "Казахстанская специфика: Kaspi/Halyk, тенге, +7 7XX, местные бренды",
    "v04": "Обфускация текста и выдача себя за публичных лиц",
    "v05": "Англоязычные казино-стримы (Twitch/Kick)",
    "v06": "Мошенничество, скам, крипто-схемы",
    "v07": "Финпирамиды HYIP / MLM",
    "v08": "Мошенничество и пирамиды на казахском языке",
    "v09": "Pump-and-dump, cash-gifting, фейковые «советники»",
}

_WORD = re.compile(r"\w+", re.UNICODE)


def _read_jsonl(path: Path) -> list:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _versions() -> list:
    files = sorted(VERSIONS_DIR.glob("v*.jsonl"))
    if not files:
        raise SystemExit(f"нет снимков датасета в {VERSIONS_DIR}")
    out = []
    for f in files:
        vid, _, commit = f.stem.partition("_")
        out.append({"id": vid, "commit": commit, "path": f})
    return out


def training_rows(path: Path, holdout_texts: set) -> tuple:
    """Строки версии без текстов отложенного набора (защита от утечки).

    Возвращает (rows, removed): removed — сколько строк совпало с held-out.
    """
    rows = _read_jsonl(path)
    keep = [r for r in rows if normalize_obfuscated(r["text"]).lower() not in holdout_texts]
    return keep, len(rows) - len(keep)


def dataset_stats(rows: list) -> dict:
    texts = [r["text"] for r in rows]
    words = [_WORD.findall(t.lower()) for t in texts]
    vocab = {w for ws in words for w in ws}
    return {
        "n": len(rows),
        "unique_texts": len(set(texts)),
        "classes": {c: sum(1 for r in rows if r["label"] == c) for c in CATEGORIES},
        "langs": dict(Counter(r.get("lang", "ru") for r in rows)),
        "avg_words": round(float(np.mean([len(w) for w in words])), 1),
        "vocab": len(vocab),
    }


def _fit(rows: list):
    X = [normalize_obfuscated(r["text"]) for r in rows]
    y = [r["label"] for r in rows]
    pipe = build_pipeline()
    t0 = time.perf_counter()
    pipe.fit(X, y)
    return pipe, round(time.perf_counter() - t0, 2)


def in_distribution_f1(rows: list, n_splits: int = 5) -> dict:
    """macro-F1 на случайном 25%-split СВОЕГО же датасета (как в train.py).

    Показывает «внутреннее» качество; на синтетике оно завышено (модель
    запоминает шаблоны), поэтому главная метрика — отложенный набор.
    """
    X = np.asarray([normalize_obfuscated(r["text"]) for r in rows], dtype=object)
    y = np.asarray([r["label"] for r in rows])
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=42)
    scores = []
    for tr, te in sss.split(X, y):
        pipe = build_pipeline()
        pipe.fit(list(X[tr]), y[tr])
        scores.append(f1_score(y[te], pipe.predict(list(X[te])), average="macro", labels=CATEGORIES))
    return {"mean": round(float(np.mean(scores)), 4), "std": round(float(np.std(scores)), 4)}


def holdout_eval(pipe, holdout: list) -> dict:
    X = [normalize_obfuscated(r["text"]) for r in holdout]
    gold = [r["label"] for r in holdout]
    proba = pipe.predict_proba(X)
    classes = list(pipe.named_steps["clf"].classes_)
    pred = [classes[i] for i in proba.argmax(axis=1)]
    clean_idx = classes.index("clean")
    risk = [int(round(100 * (1 - p[clean_idx]))) for p in proba]

    p, r, f1, support = precision_recall_fscore_support(gold, pred, labels=CATEGORIES, zero_division=0)
    threats = [i for i, g in enumerate(gold) if g != "clean"]
    cleans = [i for i, g in enumerate(gold) if g == "clean"]
    return {
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "macro_f1": round(float(f1_score(gold, pred, average="macro", labels=CATEGORIES, zero_division=0)), 4),
        # доля реальных угроз, которые модель НЕ пропустила (не отнесла к clean)
        "threat_recall": round(sum(pred[i] != "clean" for i in threats) / max(1, len(threats)), 4),
        # доля чистых постов без ложной тревоги
        "clean_specificity": round(sum(pred[i] == "clean" for i in cleans) / max(1, len(cleans)), 4),
        "per_class": {
            c: {"precision": round(float(p[k]), 4), "recall": round(float(r[k]), 4),
                "f1": round(float(f1[k]), 4), "support": int(support[k])}
            for k, c in enumerate(CATEGORIES)
        },
        "confusion_matrix": confusion_matrix(gold, pred, labels=CATEGORIES).tolist(),
        "_pred": pred,
        "_risk": risk,
    }


def run(verbose: bool = True, report=None) -> dict:
    """Прогоняет эксперимент по всем версиям, пишет compare.json и две модели."""
    holdout = _read_jsonl(HOLDOUT_PATH)
    holdout_texts = {normalize_obfuscated(r["text"]).lower() for r in holdout}
    versions = _versions()
    results = []
    pipes = {}
    for k, v in enumerate(versions):
        if report:
            report(f"Обучение на версии {v['id']} ({k + 1}/{len(versions)})",
                   int(5 + 85 * k / len(versions)))
        rows, removed = training_rows(v["path"], holdout_texts)
        pipe, fit_sec = _fit(rows)
        ev = holdout_eval(pipe, holdout)
        res = {
            "id": v["id"],
            "commit": v["commit"],
            "note": VERSION_NOTES.get(v["id"], ""),
            "stats": dataset_stats(rows),
            "removed_holdout_overlap": removed,
            "fit_sec": fit_sec,
            "in_distribution_f1": in_distribution_f1(rows),
            "holdout": {k2: v2 for k2, v2 in ev.items() if not k2.startswith("_")},
            "_pred": ev["_pred"],
            "_risk": ev["_risk"],
        }
        results.append(res)
        pipes[v["id"]] = pipe
        if verbose:
            h = res["holdout"]
            print(f"{v['id']} n={res['stats']['n']:>5}  in-dist F1={res['in_distribution_f1']['mean']:.3f}  "
                  f"holdout acc={h['accuracy']:.3f} F1={h['macro_f1']:.3f} "
                  f"threat-recall={h['threat_recall']:.3f} spec={h['clean_specificity']:.3f}")

    old, new = results[0], results[-1]
    examples = []
    for i, row in enumerate(holdout):
        o, n = old["_pred"][i], new["_pred"][i]
        if (o == row["label"]) != (n == row["label"]):
            examples.append({
                "text": row["text"],
                "gold": row["label"],
                "old_pred": o, "old_risk": old["_risk"][i],
                "new_pred": n, "new_risk": new["_risk"][i],
                "kind": "fixed" if n == row["label"] else "regressed",
            })
    examples.sort(key=lambda e: (e["kind"] != "fixed", e["gold"]))

    for r in results:
        r.pop("_pred"), r.pop("_risk")

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "FeatureUnion(tfidf-word(1,2) + tfidf-char_wb(3,5) + handcrafted) -> LogisticRegression(C=4, balanced)",
        "labels": CATEGORIES,
        "holdout": {"n": len(holdout), "classes": dict(Counter(r["label"] for r in holdout))},
        "old_id": old["id"],
        "new_id": new["id"],
        "versions": results,
        "examples": examples,
        "summary": {
            "fixed": sum(e["kind"] == "fixed" for e in examples),
            "regressed": sum(e["kind"] == "regressed" for e in examples),
        },
    }
    if report:
        report("Сохранение моделей и результатов", 95)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipes[old["id"]], "labels": CATEGORIES}, OLD_MODEL_PATH)
    joblib.dump({"pipeline": pipes[new["id"]], "labels": CATEGORIES}, NEW_MODEL_PATH)
    COMPARE_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"\nисправлено новым датасетом: {out['summary']['fixed']}, "
              f"ухудшено: {out['summary']['regressed']}  ->  {COMPARE_PATH.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    run()
