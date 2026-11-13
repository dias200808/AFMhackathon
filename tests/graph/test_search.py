"""Поиск по графу: BFS (сеть вокруг узла) и uniform-cost search (цепочка улик)."""

from app.graph import search


def _graph(edges, types):
    """edges: [(post, entity)], types: {entity_id: type}."""
    nodes = {}
    for p, e in edges:
        nodes[p] = {"id": p, "label": p, "type": "post", "risk": 80, "high_risk": True}
        nodes[e] = {"id": e, "label": e, "type": types[e], "risk": 80, "high_risk": True}
    return {"nodes": list(nodes.values()),
            "edges": [{"source": p, "target": e, "type": "contains", "weight": 1.0} for p, e in edges]}


# Игрушечный граф из пояснения во вкладке:
#   A —(бренд 1xbet)— B           короткий путь: 2 шага, стоимость 4+4 = 8
#   A —(кошелёк)— C —(telegram)— B   дешёвый путь: 4 шага, стоимость 1+1+1+1 = 4
TOY = _graph(
    [("A", "brand"), ("B", "brand"), ("A", "wallet"), ("C", "wallet"), ("C", "tg"), ("B", "tg")],
    {"brand": "betting_brand", "wallet": "crypto_wallet", "tg": "telegram"},
)


def test_bfs_layers_go_level_by_level():
    adj = search.build_adjacency(TOY, include_weak=True)
    res = search.bfs_network(adj, "A", max_depth=4)
    assert res["layers"][0] == ["A"]
    assert set(res["layers"][1]) == {"brand", "wallet"}      # соседи A
    assert set(res["layers"][2]) == {"B", "C"}               # соседи соседей
    assert res["depth"]["tg"] == 3
    assert res["order"][0] == "A"


def test_bfs_respects_max_depth():
    adj = search.build_adjacency(TOY, include_weak=True)
    res = search.bfs_network(adj, "A", max_depth=1)
    assert set(res["order"]) == {"A", "brand", "wallet"}


def test_weak_links_are_skipped_by_default():
    adj = search.build_adjacency(TOY)  # бренды — слабая связь, по умолчанию выключены
    assert "brand" not in adj
    res = search.bfs_network(adj, "A", max_depth=1)
    assert set(res["order"]) == {"A", "wallet"}


def test_generic_promo_codes_count_as_weak():
    assert search.is_generic("entity:promo_code:2026", "promo_code")
    assert search.is_generic("entity:promo_code:CODE", "promo_code")
    assert not search.is_generic("entity:promo_code:WIN777", "promo_code")
    assert search.node_cost("entity:promo_code:2026", "promo_code") == search.WEAK
    assert search.node_cost("entity:promo_code:WIN777", "promo_code") == search.MEDIUM


def test_uniform_cost_prefers_cheaper_path_over_fewer_hops():
    adj = search.build_adjacency(TOY, include_weak=True)
    res = search.uniform_cost_path(adj, "A", "B")
    assert res["found"]
    assert res["path"] == ["A", "wallet", "C", "tg", "B"]   # 4 шага, но сильные улики
    assert res["cost"] == 4                                 # а не 8 через бренд
    assert res["expanded"][0] == "A" and res["expanded"][-1] == "B"


def test_uniform_cost_reports_unreachable():
    g = _graph([("A", "w1"), ("B", "w2")], {"w1": "crypto_wallet", "w2": "crypto_wallet"})
    res = search.uniform_cost_path(search.build_adjacency(g), "A", "B")
    assert res["found"] is False and res["path"] == []


def test_network_stats():
    stats = search.network_stats(TOY, ["A", "B", "wallet"])
    assert stats["posts"] == 2 and stats["entities"] == 1
    assert stats["entity_types"] == {"crypto_wallet": 1}
    assert stats["avg_risk"] == 80 and stats["high_risk_posts"] == 2
