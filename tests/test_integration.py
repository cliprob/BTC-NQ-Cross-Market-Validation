from __future__ import annotations

from cross_market.artifacts import plot_reliability
from cross_market.events import assert_no_overlap, build_events
from cross_market.features import build_features
from cross_market.metrics import trading_metrics
from cross_market.models import choose_threshold, model_specs, probability_metrics, walk_forward_predictions
from tests.conftest import synthetic_market


def test_end_to_end_synthetic_pipeline(config, tmp_path) -> None:
    market = synthetic_market(start="2024-01-02", days=500, minutes=120)
    features = build_features(market, config)
    events = build_events(features, config, impulse_threshold=0.1)
    assert_no_overlap(events)
    assert len(events) > 100

    model_results = {}
    for spec in model_specs(config):
        predictions, importance = walk_forward_predictions(events, spec, config)
        selection = choose_threshold(predictions, config)
        assert predictions["fold"].nunique() >= 2
        assert len(importance) == len(spec.features)
        assert selection["threshold"] is not None
        selected = predictions[predictions["probability"] >= selection["threshold"]]
        metrics = trading_metrics(selected)
        assert metrics["trades"] >= config["research"]["min_validation_trades"]
        model_results[spec.name] = {"probability_metrics": probability_metrics(predictions)}

    figure = tmp_path / "reliability.png"
    plot_reliability(model_results, figure)
    assert figure.stat().st_size > 0
