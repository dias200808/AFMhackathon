"""Поиск по графу связей: сети мошенников и цепочки доказательств (AI Fundamentals, неделя 2).

Граф (app.graph.build): узлы-посты и узлы-сущности (Telegram, ссылка, кошелёк,
промокод, бренд…), ребро «пост содержит сущность». Два поста с общей сущностью
связаны через неё.

Два алгоритма неинформированного поиска:

* ``bfs_network`` — поиск в ширину (BFS) от стартового узла. Расходится слоями:
  слой 1 — соседи старта, слой 2 — соседи соседей и т.д. до ``max_depth``.
  Находит всю сеть, связанную с подозрительным постом.

* ``uniform_cost_path`` — поиск с равномерной стоимостью (uniform-cost search,
  он же Дейкстра). Каждое ребро имеет стоимость по силе связи (см. LINK_COST):
  общий кошелёк или Telegram-канал — сильная улика (дёшево), общий бренд казино —
  слабая (дорого: 1xBet упоминают тысячи несвязанных людей). Алгоритм всегда
  раскрывает узел с наименьшей накопленной стоимостью и находит самую «дешёвую»,
  то есть самую доказательную цепочку между двумя узлами.

A* здесь не используется: в этом графе нет естественной эвристики (оценки
расстояния до цели). A* с эвристикой h(n)=0 совпадает с uniform-cost search.

Оба алгоритма возвращают порядок раскрытия узлов — фронтенд проигрывает его как
анимацию, чтобы было видно, как работает поиск.
"""

import heapq
from collections import deque

# Стоимость перехода через сущность данного типа (чем меньше, тем сильнее улика).
STRONG, MEDIUM, WEAK = 1, 2, 4
LINK_COST = {
    "crypto_wallet": STRONG,
    "telegram": STRONG,
    "whatsapp": STRONG,
    "phone": STRONG,
    "url": STRONG,
    "handle": STRONG,
    "promo_code": MEDIUM,
    "payout_promise": MEDIUM,
    "casino_brand": WEAK,
    "betting_brand": WEAK,
}
WEAK_TYPES = {t for t, c in LINK_COST.items() if c == WEAK}

LINK_LABELS = {STRONG: "сильная", MEDIUM: "средняя", WEAK: "слабая"}


def link_cost(entity_type: str) -> int:
    """Стоимость ребра через сущность этого типа (неизвестный тип — средняя)."""
    return LINK_COST.get(entity_type, MEDIUM)


# «Промокоды», которые на деле общие слова или годы («Promo Code», «Бонус 500»,
# «промокод 2026») — их пишут тысячи несвязанных постов, это не улика.
_GENERIC_PROMO_WORDS = {"CODE", "CODES", "PROMO", "BONUS", "INSTANTLY", "SPINS", "POWER", "NEW"}


def is_generic(node_id: str, entity_type: str) -> bool:
    """Сущность-«пустышка»: промокод из одних цифр или общего слова."""
    if entity_type != "promo_code":
        return False
    key = node_id.rsplit(":", 1)[-1].strip().upper()
    return key.isdigit() or key in _GENERIC_PROMO_WORDS


def node_cost(node_id: str, entity_type: str) -> int:
    """Стоимость перехода через узел-сущность с учётом «пустышек»."""
    return WEAK if is_generic(node_id, entity_type) else link_cost(entity_type)


def build_adjacency(graph: dict, include_weak: bool = False) -> dict:
    """Список смежности {node_id: [(neighbor_id, cost), ...]} из графа build_graph.

    include_weak=False выкидывает слабые узлы (бренды и промокоды-«пустышки»):
    иначе любые два поста про «1xbet» оказались бы «одной сетью», хотя это не улика.
    """
    types = {n["id"]: n.get("type") for n in graph.get("nodes", [])}
    cost_of = {nid: node_cost(nid, t or "") for nid, t in types.items() if t != "post"}
    adj = {nid: [] for nid in types
           if include_weak or types[nid] == "post" or cost_of[nid] != WEAK}
    for e in graph.get("edges", []):
        s, t = e["source"], e["target"]
        if s not in adj or t not in adj:
            continue
        ent = t if types.get(s) == "post" else s
        cost = cost_of.get(ent, MEDIUM)
        adj[s].append((t, cost))
        adj[t].append((s, cost))
    for nid in adj:
        adj[nid].sort()  # детерминированный порядок обхода
    return adj


def bfs_network(adj: dict, start: str, max_depth: int = 3) -> dict:
    """Поиск в ширину от start до глубины max_depth.

    Возвращает:
      depth  — {node: номер слоя}, старт = 0;
      parent — {node: откуда пришли} (дерево обхода);
      order  — порядок посещения узлов;
      layers — [[узлы слоя 0], [узлы слоя 1], ...].
    """
    if start not in adj:
        return {"depth": {}, "parent": {}, "order": [], "layers": []}
    depth = {start: 0}
    parent = {start: None}
    order = [start]
    queue = deque([start])  # FIFO — ключевое отличие BFS от DFS
    while queue:
        u = queue.popleft()
        if depth[u] >= max_depth:
            continue
        for v, _cost in adj[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                parent[v] = u
                order.append(v)
                queue.append(v)
    layers = [[] for _ in range(max(depth.values()) + 1)]
    for node in order:
        layers[depth[node]].append(node)
    return {"depth": depth, "parent": parent, "order": order, "layers": layers}


def uniform_cost_path(adj: dict, start: str, goal: str) -> dict:
    """Uniform-cost search: самый дешёвый путь start -> goal.

    Возвращает found, path (узлы), cost (сумма стоимостей), expanded (порядок
    раскрытия узлов, т.е. извлечения из очереди с приоритетом).
    """
    empty = {"found": False, "path": [], "cost": None, "expanded": []}
    if start not in adj or goal not in adj:
        return empty
    tie = 0  # разрыв ничьих в куче: при равной стоимости — кто раньше добавлен
    frontier = [(0, tie, start)]
    best = {start: 0}
    parent = {start: None}
    expanded = []
    done = set()
    while frontier:
        g, _, u = heapq.heappop(frontier)
        if u in done:
            continue  # устаревшая запись: до u уже нашли путь дешевле
        done.add(u)
        expanded.append(u)
        if u == goal:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return {"found": True, "path": path[::-1], "cost": g, "expanded": expanded}
        for v, cost in adj[u]:
            ng = g + cost
            if v not in done and ng < best.get(v, float("inf")):
                best[v] = ng
                parent[v] = u
                tie += 1
                heapq.heappush(frontier, (ng, tie, v))
    return {**empty, "expanded": expanded}


def network_stats(graph: dict, node_ids) -> dict:
    """Сводка по найденной сети: сколько постов/сущностей, риск."""
    by_id = {n["id"]: n for n in graph.get("nodes", [])}
    nodes = [by_id[i] for i in node_ids if i in by_id]
    posts = [n for n in nodes if n.get("type") == "post"]
    ents = [n for n in nodes if n.get("type") != "post"]
    by_type = {}
    for n in ents:
        by_type[n["type"]] = by_type.get(n["type"], 0) + 1
    risks = [n.get("risk", 0) for n in posts]
    return {
        "posts": len(posts),
        "entities": len(ents),
        "entity_types": by_type,
        "avg_risk": round(sum(risks) / len(risks)) if risks else 0,
        "max_risk": max(risks) if risks else 0,
        "high_risk_posts": sum(1 for n in posts if n.get("high_risk")),
    }
