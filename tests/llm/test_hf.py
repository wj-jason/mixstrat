import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixstrat.game.config import GameConfig  # noqa: E402
from mixstrat.game.tree import GameTree  # noqa: E402
from mixstrat.llm.hf import DEFAULT_MODEL, letter_logprobs_from_row  # noqa: E402
from mixstrat.llm.policy import query_profile  # noqa: E402
from mixstrat.llm.prompts import build_prompt  # noqa: E402


def test_letter_logprobs_add_variants_and_renormalize() -> None:
    probs = torch.tensor([0.1, 0.2, 0.3, 0.05, 0.35])  # tokens 0..4
    out = letter_logprobs_from_row(probs.log(), [[0, 2], [1]])  # A = tokens 0+2, B = token 1
    assert float(out.legal_mass) == pytest.approx(0.6)
    assert out.logprobs.exp().tolist() == pytest.approx([0.4 / 0.6, 0.2 / 0.6])


def test_letter_logprobs_are_differentiable() -> None:
    logits = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
    out = letter_logprobs_from_row(torch.log_softmax(logits, dim=0), [[0], [1]])
    out.logprobs[0].backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0


def _model_cached() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(DEFAULT_MODEL, "config.json"), str)


@pytest.mark.gpu
@pytest.mark.skipif(not _model_cached(), reason=f"needs CUDA and cached {DEFAULT_MODEL} weights")
def test_hf_backend_scores_kuhn_and_trains() -> None:
    from peft import LoraConfig, get_peft_model

    from mixstrat.llm.hf import HFBackend

    tree = GameTree(GameConfig(name="kuhn"))
    backend = HFBackend.load()
    result = query_profile(tree, backend)
    assert result.min_legal_mass > 0.9
    for probs in result.profile.probs.values():
        assert np.allclose(probs.sum(axis=1), 1.0)

    lora = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    backend.model = get_peft_model(backend.model, lora)
    backend.model.train()
    prompt = build_prompt(tree, tree.infosets[0])
    (out,) = backend.letter_logprobs([(prompt.messages, prompt.letters)])
    assert math.isclose(float(out.logprobs.exp().sum()), 1.0, rel_tol=1e-5)
    out.logprobs[0].backward()
    grads = [p.grad for p in backend.model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    backend.close()
