"""HTTP-роуты графа связей (F5) — подключаются авто-роутером main.py (A2).

GET /api/graph?post_id=&min_risk=
  - post_id задан      -> эго-сеть этого поста (его сущности + co-posts);
  - post_id отсутствует -> весь граф revealed-постов с риском >= min_risk.

Поиск по графу (app.graph.search):
GET /api/graph/network?node=&depth=&weak=&min_risk=   — BFS: сеть вокруг узла
GET /api/graph/path?source=&target=&weak=&min_risk=   — uniform-cost: цепочка между узлами
GET /api/graph/search-examples?min_risk=              — готовые старт/пара для демо

Соединение БД берётся из request.app.state.db (одно соединение из lifespan,
§0.2); функции построения вызываются с conn=этим соединением (не закрываем).
"""

from fastapi import APIRouter, HTTPException, Query, Request

from app.graph import search
from app.graph.build import build_ego_graph, build_graph

router = APIRouter()


def _whole_graph(conn, min_risk: int) -> dict:
    rows = conn.execute(
        "SELECT s.post_id FROM scores s "
        "JOIN posts p ON p.id = s.post_id "
        "WHERE p.revealed = 1 AND s.risk >= ? "
        "ORDER BY s.risk DESC",
        (min_risk,),
    ).fetchall()
    return build_graph([r["post_id"] for r in rows], conn=conn)


@router.get("/api/graph")
def api_graph(request: Request, post_id: "str | None" = None, min_risk: int = 0):
    conn = request.app.state.db
    if post_id:
        return build_ego_graph(post_id, conn=conn)
    return _whole_graph(conn, min_risk)


_NOT_FOUND = "узел не найден в графе (или это слабая связь при выключенном переключателе)"


def _node_brief(by_id: dict, nid: str, meta: "dict | None" = None) -> dict:
    n = by_id.get(nid, {})
    out = {"id": nid, "label": n.get("label", nid), "type": n.get("type"),
           "risk": n.get("risk", 0)}
    if meta and nid in meta:
        out.update(meta[nid])
    return out


def _edge_cost(by_id: dict, a: str, b: str) -> int:
    ent = b if by_id.get(a, {}).get("type") == "post" else a
    return search.node_cost(ent, by_id.get(ent, {}).get("type", ""))


def _post_meta(conn, node_ids) -> dict:
    """Подписи постов для понятных подписей узлов (вместо технического id)."""
    pids = [n[5:] for n in node_ids if n.startswith("post:")]
    if not pids:
        return {}
    q = ",".join("?" for _ in pids)
    rows = conn.execute(
        f"SELECT id, platform, author_handle, caption FROM posts WHERE id IN ({q})", tuple(pids)
    ).fetchall()
    out = {}
    for r in rows:
        cap = " ".join((r["caption"] or "").split())
        out["post:" + r["id"]] = {
            "author": r["author_handle"] or "",
            "platform": r["platform"] or "",
            "caption": cap[:90] + ("…" if len(cap) > 90 else ""),
        }
    return out


@router.get("/api/graph/network")
def api_graph_network(request: Request, node: str, depth: int = Query(3, ge=1, le=6),
                      weak: bool = False, min_risk: int = 0):
    conn = request.app.state.db
    graph = _whole_graph(conn, min_risk)
    adj = search.build_adjacency(graph, include_weak=weak)
    if node not in adj:
        raise HTTPException(status_code=404, detail=_NOT_FOUND)
    res = search.bfs_network(adj, node, max_depth=depth)
    by_id = {n["id"]: n for n in graph["nodes"]}
    meta = _post_meta(conn, res["order"])
    nodes = [{**_node_brief(by_id, nid, meta), "depth": res["depth"][nid]} for nid in res["order"]]
    tree = [{"source": p, "target": c, "cost": _edge_cost(by_id, p, c)}
            for c, p in res["parent"].items() if p is not None]
    return {
        "algorithm": "bfs",
        "start": node,
        "max_depth": depth,
        "nodes": nodes,
        "tree_edges": tree,
        "order": res["order"],
        "layers": res["layers"],
        "stats": search.network_stats(graph, res["order"]),
    }


@router.get("/api/graph/path")
def api_graph_path(request: Request, source: str, target: str, weak: bool = False,
                   min_risk: int = 0):
    conn = request.app.state.db
    graph = _whole_graph(conn, min_risk)
    adj = search.build_adjacency(graph, include_weak=weak)
    for nid in (source, target):
        if nid not in adj:
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
    res = search.uniform_cost_path(adj, source, target)
    by_id = {n["id"]: n for n in graph["nodes"]}
    meta = _post_meta(conn, res["path"])
    steps = []
    for a, b in zip(res["path"], res["path"][1:]):
        c = _edge_cost(by_id, a, b)
        steps.append({"source": a, "target": b, "cost": c, "strength": search.LINK_LABELS[c]})
    return {
        "algorithm": "uniform_cost",
        "found": res["found"],
        "source": source,
        "target": target,
        "path": [_node_brief(by_id, nid, meta) for nid in res["path"]],
        "steps": steps,
        "cost": res["cost"],
        "hops": max(0, len(res["path"]) - 1),
        "expanded": res["expanded"],
        "expanded_count": len(res["expanded"]),
    }


@router.get("/api/graph/search-examples")
def api_graph_search_examples(request: Request, min_risk: int = 0):
    """Готовые примеры для демо.

    network_start — самая связанная сущность крупнейшей сети на СИЛЬНЫХ связях
    (общие ссылки/каналы/кошельки). path_source/path_target — посты из двух
    крупнейших сетей: без слабых связей пути между ними нет, со слабыми он есть,
    но дорогой — наглядно, зачем у рёбер стоимость.
    """
    conn = request.app.state.db
    graph = _whole_graph(conn, min_risk)
    adj = search.build_adjacency(graph, include_weak=False)
    strong = {k: [(v, w) for v, w in vs if w == search.STRONG] for k, vs in adj.items()}
    by_id = {n["id"]: n for n in graph["nodes"]}
    comps = []
    seen = set()
    for nid in sorted(strong):
        if nid in seen:
            continue
        comp = search.bfs_network(strong, nid, max_depth=10**6)["order"]
        seen.update(comp)
        posts = [x for x in comp if by_id[x]["type"] == "post"]
        if len(posts) >= 2:
            comps.append((len(posts), len(comp) - len(posts), comp, posts))
    comps.sort(key=lambda c: (c[0], c[1]), reverse=True)
    out = {"network_start": None, "network_posts": 0, "path_source": None, "path_target": None}
    if not comps:
        return out
    comp = comps[0][2]
    ents = [x for x in comp if by_id[x]["type"] != "post"]
    start = max(ents, key=lambda x: (len(strong[x]), x)) if ents else comp[0]
    out["network_start"] = _node_brief(by_id, start)
    out["network_posts"] = comps[0][0]
    if len(comps) >= 2:
        a, b = sorted(comps[0][3])[0], sorted(comps[1][3])[0]
        meta = _post_meta(conn, [a, b])
        out["path_source"] = _node_brief(by_id, a, meta)
        out["path_target"] = _node_brief(by_id, b, meta)
    return out
