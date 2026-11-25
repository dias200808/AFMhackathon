# KÓZ · AI Media Watch — Technical Report

**Course:** AI Fundamentals — Final Project
**Repository:** https://github.com/dias200808/AFMhackathon
**Demo video:** [`docs/final/demo_video.mp4`](demo_video.mp4) (2 min 48 s)

---

## 1. Problem and objectives

Illegal online casinos, unlicensed bookmakers and financial pyramids advertise on YouTube,
Telegram, TikTok and Instagram in Kazakhstan, in Russian and Kazakh. The Agency for Financial
Monitoring (AFM) cannot check thousands of posts and videos by hand, and blocking one account
does not help when the same scheme runs through many accounts.

**Objectives**

1. Turn a social-media post or video into text, including speech and text shown on screen.
2. Score every post: category (gambling / pyramid / fraud / clean) and risk 0–100, with an explanation.
3. Find the network of accounts, links and wallets behind a risky post, and the chain of evidence between two accounts.
4. Keep a human in the loop: the system prioritises, an analyst decides.

## 2. System overview

```
post / video
  │  Topic 3 — Neural networks (pre-trained)
  ├─ Whisper ........ speech → text
  ├─ EasyOCR ........ text on video frames → text
  └─ CLIP ........... frame → visual markers (casino, betting slip, cash, …)
  ▼
combined text ── normalisation (anti-obfuscation) + 14 hand-made features
  │  Topic 2 — Classical machine learning
  └─ TF-IDF (words + characters) → Logistic Regression → category + risk 0–100
  ▼
decision thresholds: ≥70 escalate · 40–69 review · <40 clear  → analyst queue
  │  Topic 1 — Search
  └─ graph of posts ↔ entities → BFS (whole network) · Uniform-cost search (evidence chain)
```

Each stage uses the output of the previous one: the neural networks produce the text that the
classifier scores; the classifier’s risk score is attached to graph nodes; graph search then
groups risky posts into networks and explains how two accounts are connected.

**Stack:** Python 3.12, FastAPI, SQLite, scikit-learn, faster-whisper, EasyOCR, open-CLIP,
yt-dlp + ffmpeg, Alpine.js + Chart.js + vis-network (no-build web UI), pytest (587 tests).

## 3. Topic 1 — Search: BFS and uniform-cost search (Week 2)

Code: `app/graph/search.py`, API in `app/graph/routes.py`, UI in the **Graph** tab.

**The graph.** Nodes are posts and the entities extracted from them (URLs, Telegram channels,
accounts, crypto wallets, phones, promo codes, casino brands). An edge means “post contains
entity”. Two posts that share a wallet are connected through it.

**Edge costs = strength of evidence.**

| Link type | Examples | Cost |
|---|---|---|
| strong | URL, Telegram channel, account, crypto wallet, phone | 1 |
| medium | unique promo code, payout promise | 2 |
| weak | casino/bookmaker brand; generic “promo codes” such as “Promo Code” or “2025” | 4 |

A shared brand is weak evidence because thousands of unrelated people mention 1xBet. Weak links
are excluded by default and can be switched on.

**BFS (breadth-first search).** A FIFO queue explores the graph layer by layer: the start node,
its neighbours, their neighbours, up to a depth limit. Here BFS is used as a *traversal*: the goal
is to collect the whole network connected to a suspicious post. Because near nodes are visited
first, the layer number equals the shortest number of steps from the start, and a depth limit
returns the closest network. Complexity O(V + E).
*Why BFS and not DFS:* DFS can follow one long branch and does not give shortest distances.

**Uniform-cost search (UCS).** A priority queue always expands the node with the lowest total
path cost, so the first time the goal is expanded, the path is the cheapest one — the chain with
the strongest evidence. UCS may prefer a longer path of strong links over a short path of weak
links (unit test `test_uniform_cost_prefers_cheaper_path_over_fewer_hops`: 4 strong steps,
cost 4, beat 2 weak steps, cost 8). Complexity O(E log V).
*Why not A\*:* A\* needs a heuristic (an estimate of the distance to the goal). This graph has no
natural one; A\* with h(n) = 0 is exactly uniform-cost search.

**How it is shown.** The UI animates the search: BFS layers light up one after another; for UCS,
expanded nodes turn yellow in expansion order and the final path is drawn in red with the
strength of each link. A “How it works” panel explains both algorithms.

**Result on live data (recorded in the demo video, 25 Nov 2026):**

- BFS from the Telegram account **@melbetsport** found **15 posts** connected through
  **3 shared entities** (the account and 2 short links).
- UCS between **@melbetsport** and **@mostbet_casino**: with strong and medium links only —
  **no path** (18 nodes expanded), so these are separate networks; with weak links — a path of
  **4 steps, cost 16**, every link a shared brand or generic word, i.e. coincidence, not proof.

Examples change as the system keeps discovering new posts; the “Example” button always picks
the largest current network.

## 4. Topic 2 — Classical machine learning: TF-IDF + logistic regression (Week 8)

Code: `app/model/train.py`, `app/model/features.py`, `app/model/classifier.py`.

**Features.**

- *Normalisation* undoes scam tricks: spaced letters (“к а з и н о”), Latin look-alike letters
  inside Cyrillic words, digits instead of letters.
- *TF-IDF on words* (1–2-grams) and *on characters* (3–5-grams). TF-IDF (term frequency –
  inverse document frequency) gives a word a high weight when it is frequent in this post but
  rare in others. Character n-grams handle Kazakh endings and obfuscation (“казик”, “kазино”).
- *14 hand-made features* for Kazakhstan-specific signals: Kaspi/Halyk transfers, tenge
  amounts, +7 7XX phones, local bookmaker brands, urgency, referral, crypto.

**Model.** Logistic regression (`C=4`, `class_weight="balanced"`) gives every feature a weight
for every class; the weighted sums are turned into class probabilities by softmax.
Risk = 100 × (1 − P(clean)). Training takes under one second.

*Why logistic regression:* small dataset (~2,000 rows), fast, and explainable — the weights of
the fired signals are shown to the analyst as reasons. It is equivalent to a one-layer neural
network with a softmax output.

**Decision rule (rational decision).** Risk ≥ 70 → escalate, 40–69 → review, < 40 → clear.
The review threshold is below 50 because missing a threat costs more than a false alarm.

**Evaluation.** A fixed held-out set of 106 realistic posts that is never used in training;
overlaps with training data are removed automatically (tested).

| Metric | Value |
|---|---|
| Accuracy | **93.4%** (99 / 106) |
| Macro-F1 | **94.7%** |
| Threats caught (dangerous posts not labelled clean) | **95.8%** (68 / 71) |
| No false alarms (clean posts kept clean) | **100%** (35 / 35) |

**Effect of data.** With the same model, training on the original template dataset
(520 rows) gave 54.7% accuracy; the expanded dataset (1,952 rows: hard negatives, Kazakh texts,
new fraud schemes) gave 93.4%. The original dataset scored F1 0.997 on its own split but 0.639
on real posts — it had memorised templates. (Tab **Old vs new**, `scripts/compare_datasets.py`.)

## 5. Topic 3 — Neural networks: Whisper, EasyOCR, CLIP (Weeks 9–10)

Code: `app/extractors/audio.py`, `ocr.py`, `visual.py`, orchestration in `pipeline.py`.
These are **pre-trained** networks integrated into the pipeline, not trained by us.

| Model | What it does | Inside |
|---|---|---|
| **Whisper** (faster-whisper, `small`) | speech → text, detects language | encoder–decoder transformer |
| **EasyOCR** (ru + en) | reads text on video frames | CNN detector + CRNN recogniser |
| **CLIP** (ViT-B-32, LAION-2B) | compares a frame with text prompts → visual markers | vision transformer + text transformer, zero-shot |

**How CLIP is used.** Every frame is compared with gambling prompts (“a casino interior with
slot machines”, “a betting slip”, …) and neutral prompts (“a person talking to the camera”,
“a news broadcast”, …). A gambling label is emitted only if it beats the best neutral prompt by
a margin, which reduces false alarms.

*Why CLIP and not YOLO:* YOLO detects a fixed list of object classes (person, car, …) and does
not know “roulette” or “slot machine” without a custom labelled dataset. CLIP works zero-shot
with any text description.

**Live example** (YouTube video, first 90 seconds, Apple-silicon CPU):

| Stage | Output | Time |
|---|---|---|
| download (yt-dlp) | 90 s clip + 4 frames | ~7 s |
| Whisper | “Приветствуем всех любителей ставок… промокод… +100% на депозит…” (ru) | ~23 s |
| EasyOCR | “18+ ПРОМОКОД: VIPMBET … MELBET … ССЫЛКА НА САЙТ MELBET С БОНУСОМ” | ~9 s |
| CLIP | betting_slip 0.997, casino 0.74 | ~5 s |
| classifier | **gambling, risk 100 → escalate**, with reasons | <1 s |

**Bug found and fixed during integration.** faster-whisper reads audio through PyAV; PyAV 16+
removed an argument it uses, so Whisper silently returned an empty transcript. The dependency is
now pinned (`av>=11,<16` in `requirements-extractors.txt`).

## 6. Results summary

- End-to-end pipeline works on real videos: speech, on-screen text and scene are all extracted
  and scored in about 45 s per 90-second clip on a laptop CPU.
- Classifier: 93.4% accuracy and 95.8% of threats caught on held-out data, no false alarms.
- Graph search turns single flags into networks and separates strong evidence from coincidence.
- 587 automated tests pass (`python -m pytest tests -q`).

## 7. Limitations and future work

- **Data:** training data is synthetic; the held-out set is small (106 posts, only 3 gambling)
  and guided the dataset expansion, so the score may be optimistic. Next: real labelled posts
  and a fresh final test set.
- **Extraction noise:** the regex extractor sometimes treats years and generic words as promo
  codes; graph search already treats them as weak links, the extractor itself should be fixed.
- **Search:** add goal-based BFS (“nearest crypto wallet from this post”) and A\* if a useful
  heuristic appears (e.g. time or geography).
- **Speed:** Whisper on Apple MPS or a GPU would be 2× faster.
- **Scale:** PostgreSQL and a job queue instead of SQLite for production.

## 8. Ethics and AI safety

- Human-in-the-loop: the system only prioritises; it never blocks or accuses automatically.
- Every flag is explained by the signals that fired.
- Only public posts are processed; personal data is not profiled.
- Licensed operators are marked as legal in Kazakhstan, so legal businesses are not escalated as
  illegal.

## 9. How to run

```bash
git clone https://github.com/dias200808/AFMhackathon.git && cd AFMhackathon
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-extractors.txt   # Whisper, EasyOCR, CLIP (~2.5 GB with models)
.venv/bin/python -m uvicorn app.main:app --port 8000            # open http://127.0.0.1:8000
```

Tests: `.venv/bin/python -m pytest tests -q`. Video analysis also needs `ffmpeg` on the PATH.
