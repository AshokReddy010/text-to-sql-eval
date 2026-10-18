import pytest

from sqleval.scoring import connect
from sqleval.warehouse import build


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("wh") / "warehouse.duckdb"
    build(p)
    return p


@pytest.fixture(scope="session")
def con(db_path):
    c = connect(str(db_path))
    yield c
    c.close()
