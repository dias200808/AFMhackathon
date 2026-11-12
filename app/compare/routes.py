"""Роуты вкладки «Старый vs новый датасет» (авто-роутер, main.py не правим).

GET  /api/compare          — результаты эксперимента (compare.json)
POST /api/compare/predict  — {"text"} -> предсказания обеих моделей рядом
POST /api/compare/run      — перезапустить эксперимент в фоне -> {job_id}
"""

import json
import threading

from fastapi import APIRouter, Body, HTTPException

import app.jobs.worker as jobs
from app.config import CATEGORIES
from app.model.normalize import normalize_obfuscated
from scripts import compare_datasets as exp

router = APIRouter()

_models: dict = {}
_lock = threading.Lock()


def _load_models() -> dict:
    with _lock:
        if not _models:
            import joblib
            # Локальные артефакты, созданные scripts/compare_datasets.py в этом репо.
            _models["old"] = joblib.load(exp.OLD_MODEL_PATH)["pipeline"]
            _models["new"] = joblib.load(exp.NEW_MODEL_PATH)["pipeline"]
        return _models


def _predict(pipe, text: str) -> dict:
    proba = pipe.predict_proba([normalize_obfuscated(text)])[0]
    classes = list(pipe.named_steps["clf"].classes_)
    probs = {c: round(float(proba[classes.index(c)]), 4) for c in CATEGORIES}
    category = max(probs, key=probs.get)
    return {
        "category": category,
        "risk": int(round(100 * (1 - probs["clean"]))),
        "probs": probs,
    }


@router.get("/api/compare")
def get_compare() -> dict:
    if not exp.COMPARE_PATH.exists():
        raise HTTPException(status_code=404,
                            detail="Эксперимент ещё не запускался: python -m scripts.compare_datasets")
    return json.loads(exp.COMPARE_PATH.read_text(encoding="utf-8"))


@router.post("/api/compare/predict")
def compare_predict(payload: dict = Body(...)) -> dict:
    text = (payload or {}).get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="поле 'text' обязательно")
    if not (exp.OLD_MODEL_PATH.exists() and exp.NEW_MODEL_PATH.exists()):
        raise HTTPException(status_code=404, detail="модели сравнения не обучены")
    m = _load_models()
    return {"old": _predict(m["old"], text), "new": _predict(m["new"], text)}


@router.post("/api/compare/run")
def compare_run() -> dict:
    def _run(report):
        out = exp.run(verbose=False, report=report)
        with _lock:
            _models.clear()  # подхватить свежие модели при следующем predict
        return {"summary": out["summary"], "generated_at": out["generated_at"]}

    return {"job_id": jobs.run_job("compare", _run), "status": "queued"}
