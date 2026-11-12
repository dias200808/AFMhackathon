"""Вкладка «Старый vs новый датасет»: API и инварианты эксперимента."""

from fastapi.testclient import TestClient

from app.main import app
from scripts import compare_datasets as exp


def test_versions_old_is_smallest_new_is_current_dataset():
    vs = exp._versions()
    assert len(vs) >= 2
    old_n = len(exp._read_jsonl(vs[0]["path"]))
    new_n = len(exp._read_jsonl(vs[-1]["path"]))
    assert old_n < new_n


def test_training_rows_exclude_holdout():
    from app.model.normalize import normalize_obfuscated as norm
    holdout = {norm(r["text"]).lower() for r in exp._read_jsonl(exp.HOLDOUT_PATH)}
    for v in exp._versions():
        rows, _ = exp.training_rows(v["path"], holdout)
        assert not any(norm(r["text"]).lower() in holdout for r in rows), f"утечка в {v['id']}"


def test_get_compare_shape():
    with TestClient(app) as client:
        r = client.get("/api/compare")
        assert r.status_code == 200
        d = r.json()
        assert d["old_id"] == d["versions"][0]["id"]
        assert d["new_id"] == d["versions"][-1]["id"]
        for v in d["versions"]:
            h = v["holdout"]
            assert 0 <= h["accuracy"] <= 1
            assert sum(map(sum, h["confusion_matrix"])) == d["holdout"]["n"]


def test_predict_returns_both_models():
    with TestClient(app) as client:
        r = client.post("/api/compare/predict", json={"text": "Промокод на казино 1xbet, первый депозит удвоим"})
        assert r.status_code == 200
        d = r.json()
        for side in ("old", "new"):
            assert d[side]["category"] in d[side]["probs"]
            assert 0 <= d[side]["risk"] <= 100


def test_predict_requires_text():
    with TestClient(app) as client:
        assert client.post("/api/compare/predict", json={}).status_code == 400
        assert client.post("/api/compare/predict", json={"text": "  "}).status_code == 400
