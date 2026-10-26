import os

import numpy as np
import pytest

from mixstrat.game.config import GameConfig
from mixstrat.game.tree import GameTree
from mixstrat.llm.policy import query_profile

httpx = pytest.importorskip("httpx")
from mixstrat.llm.ollama import DEFAULT_HOST, OllamaBackend  # noqa: E402

MODEL = os.environ.get("MIXSTRAT_OLLAMA_MODEL", "qwen3:1.7b")


def _model_available() -> bool:
    try:
        tags = httpx.get(f"{DEFAULT_HOST}/api/tags", timeout=2).json()
    except httpx.HTTPError:
        return False
    return any(m["name"] == MODEL for m in tags.get("models", []))


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(not _model_available(), reason=f"Ollama with {MODEL} not available"),
]


def test_kuhn_profile_from_ollama() -> None:
    tree = GameTree(GameConfig(name="kuhn"))
    with OllamaBackend(MODEL) as backend:
        result = query_profile(tree, backend)
    assert len(result.queries) == len(tree.infosets)
    assert result.min_legal_mass > 0.5
    for probs in result.profile.probs.values():
        assert np.allclose(probs.sum(axis=1), 1.0)
