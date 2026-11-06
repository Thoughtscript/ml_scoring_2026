"""Run every stage in order in a single process: data -> train -> tests.
(docker compose runs each stage as its own container instead; see docker-compose.yml.)"""
import pytest

from common import generate_data
from techniques import classification, regression

STAGES = [
    generate_data.main,
    classification.main,
    regression.main,
]

if __name__ == "__main__":
    for fn in STAGES:
        fn()
    raise SystemExit(pytest.main(["-v", "-s", "tests"]))
