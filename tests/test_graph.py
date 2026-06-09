from app.graph import route


def test_route_takes_good_path_when_chunks_survive():
    assert route(["doc"], 0, 2) == "generate"


def test_route_refines_while_budget_remains():
    assert route([], 0, 2) == "refine"
    assert route([], 1, 2) == "refine"


def test_route_gives_up_when_loops_exhausted():
    assert route([], 2, 2) == "give_up"
