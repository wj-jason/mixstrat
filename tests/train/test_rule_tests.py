import numpy as np
import pytest

from mixstrat.train.rule_tests import (
    _reversed,
    equilibria,
    hypothesis_fit,
    one_change,
    probe_configs,
)


@pytest.fixture(scope="module")
def solved():
    return equilibria(2000)


def _model_from(eq, key_of):
    """A fake model: on every probe it plays eq[key_of(probe)], and on the reference
    games it plays like classic Kuhn for J/Q/K."""
    model = {key: eq[key_of(key)] for key in eq}
    classic = eq["jqk/asc/high"]
    lowball = eq["jqk/asc/low"]
    model["ref/kuhn_classic"] = classic
    # 13-card references: put the 3-card strategy on the cards the probes look up
    # (2, 3, 4 = ranks 0-2; J, Q, K = ranks 9-11); other cards are never read.
    for ref, src in (("ref/kuhn13", classic), ("ref/kuhn13_lowball", lowball)):
        model[ref] = {}
        for (p, c, h), v in src.items():
            model[ref][(p, c, h)] = v
            model[ref][(p, c + 9, h)] = v
    return model


def test_a_rule_reader_fits_rules_and_follows_every_line(solved) -> None:
    _, eq = solved
    model = _model_from(eq, lambda k: k)
    base = {k: {kk: np.full(2, 0.5) for kk in v} for k, v in model.items()}
    fits = hypothesis_fit(model, base, eq)
    assert all(min(v, key=v.get) == "rules" for v in fits.values())
    t2 = one_change(model, eq)
    assert t2["reverse listing"]["sensitivity"] == pytest.approx(1)
    assert t2["flip showdown"]["alignment"] == pytest.approx(1)
    assert t2["rename"]["model_shift"] == pytest.approx(0)


def test_a_position_player_ignores_the_showdown_line(solved) -> None:
    _, eq = solved
    model = _model_from(eq, lambda k: k.replace("/low", "/high"))
    t2 = one_change(model, eq)
    assert t2["flip showdown"]["sensitivity"] == pytest.approx(0)
    assert t2["reverse listing"]["sensitivity"] == pytest.approx(1)


def test_reversed_matches_card_names() -> None:
    cfgs = probe_configs()
    asc, rev = cfgs["jqk/asc/high"].card_names, cfgs["jqk/rev/high"].card_names
    for i in range(3):
        assert asc[i] == rev[_reversed((0, i, ()))[1]]
