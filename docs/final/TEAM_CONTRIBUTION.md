# Team Contribution Statement

**Project:** KÓZ · AI Media Watch — AI Fundamentals Final Project

> ⚠️ Fill in before submission: team members for this course and the exact split of work.

## Team (AI Fundamentals)

| Member | Role in the final project |
|---|---|
| Dias [Surname] | [e.g. individual project / team lead] |
| [Name, if any] | [role] |

## Origin of the code base

The base system KÓZ (classifier, extractors, feed, monitoring, PDF reports) was built during the
**AFM AI Hackathon 2026** by our hackathon team. The Git history shows the base commits
(authored by Adlet Seitenov). The work below was added for this course.

## Contributions made for the final project

| Contribution | Topic | Files |
|---|---|---|
| Graph search: BFS (network around a node) and uniform-cost search (evidence chain), edge costs by evidence strength, demo examples | Search | `app/graph/search.py`, `app/graph/routes.py` |
| Graph-tab UI: “How it works” panel, search panel, animated BFS layers and UCS expansion, evidence chain | Search | `web/index.html`, `web/app.js` |
| Old-vs-new dataset experiment (10 dataset versions, leakage check) and the comparison tab | Classical ML | `scripts/compare_datasets.py`, `app/compare/` |
| Enabling the neural-network pipeline (Whisper, EasyOCR, CLIP), fixing the Whisper/PyAV bug, testing on real videos | Neural networks | `requirements-extractors.txt` |
| Tests for the new code (16 tests) | — | `tests/graph/test_search.py`, `tests/graph/test_routes.py`, `tests/compare/` |
| Technical report, demo video, AI usage declaration | — | `docs/final/` |

## Declaration

Each team member can explain the whole project and answer questions about all three AI topics.

Signed: ____________________   Date: ____________
