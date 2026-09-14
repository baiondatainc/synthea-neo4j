import pytest

from harness import CFG, Graph, Results, load_questions, warm_up


def pytest_addoption(parser):
    parser.addoption("--no-api", action="store_true", default=False,
                     help="Skip the models; only validate that ground-truth Cypher runs.")
    parser.addoption("--category", action="store", default=None, help="Run one category only.")
    parser.addoption("--ids", action="store", default=None, help="Comma-separated ids, e.g. Q01,Q19,Q34")


def pytest_generate_tests(metafunc):
    if "q" in metafunc.fixturenames:
        qs = load_questions()
        cat = metafunc.config.getoption("--category")
        ids = metafunc.config.getoption("--ids")
        if cat:
            qs = [q for q in qs if q.category == cat]
        if ids:
            wanted = {i.strip() for i in ids.split(",")}
            qs = [q for q in qs if q.id in wanted]
        metafunc.parametrize("q", qs, ids=[f"{q.id}-{q.category}" for q in qs])


@pytest.fixture(scope="session")
def graph():
    g = Graph(CFG)
    try:
        g.verify()
    except Exception as e:  # noqa: BLE001
        g.close()
        pytest.exit(f"Cannot connect to Neo4j at {CFG.neo4j_uri} as {CFG.neo4j_user}: {e}\n"
                    f"Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD in .env and retry.", returncode=2)
    yield g
    g.close()


@pytest.fixture(scope="session")
def no_api(request):
    return request.config.getoption("--no-api")


@pytest.fixture(scope="session", autouse=True)
def models_ready(request, no_api):
    if no_api:
        return
    try:
        warm_up(CFG)
    except Exception as e:  # noqa: BLE001
        pytest.exit(f"Model backend not reachable ({CFG.mode}): {e}", returncode=2)


@pytest.fixture(scope="session")
def results():
    r = Results()
    yield r
    r.write_summary()
    print(f"\nSummary written to {r.summary}")