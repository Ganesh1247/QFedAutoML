"""
[IMPLEMENTED] Unit and integration tests for Explainability & Trust subsystems:
SHAP Explainer, LIME Local Surrogate, Transformer Attention Visualizer, and Report Generator.
"""
import torch

from backend.automl.preprocessing import (
    load_sensor_timeseries_dataset,
    load_starter_tabular_dataset,
)
from backend.explainability.attention_visualizer import attention_visualizer
from backend.explainability.lime_explainer import lime_explainer
from backend.explainability.report_generator import report_generator
from backend.explainability.shap_explainer import shap_explainer
from backend.models.classical_models import XGBoostModel
from backend.models.transformer_model import TimeSeriesTransformerNN


def test_shap_explainer_global_and_local():
    """Verify SHAP global rankings and local instance attributions."""
    splits = load_starter_tabular_dataset()
    model = XGBoostModel(n_estimators=30, max_depth=3, random_state=42)
    model.fit(splits.X_train, splits.y_train)

    # 1. Global SHAP
    global_res = shap_explainer.explain_global(
        model=model,
        X=splits.X_val,
        feature_names=splits.feature_names,
        max_samples=50
    )
    assert global_res["explainer"] is not None
    assert len(global_res["rankings"]) == len(splits.feature_names)
    assert global_res["rankings"][0]["rank"] == 1
    assert global_res["rankings"][0]["mean_abs_shap"] >= global_res["rankings"][-1]["mean_abs_shap"]

    # 2. Local Instance SHAP
    sample_inst = splits.X_val[0]
    local_res = shap_explainer.explain_instance(
        model=model,
        instance=sample_inst,
        background_data=splits.X_train,
        feature_names=splits.feature_names
    )
    assert "predicted_value" in local_res
    assert len(local_res["feature_contributions"]) == len(splits.feature_names)
    assert abs(local_res["feature_contributions"][0]["shap_value"]) >= abs(local_res["feature_contributions"][-1]["shap_value"])


def test_lime_local_surrogate_explainer():
    """Verify LIME surrogate linear model perturbation and explanation."""
    splits = load_starter_tabular_dataset()
    model = XGBoostModel(n_estimators=30, max_depth=3, random_state=42)
    model.fit(splits.X_train, splits.y_train)

    sample_inst = splits.X_val[0]
    lime_res = lime_explainer.explain_instance(
        model=model,
        instance=sample_inst,
        training_data=splits.X_train,
        feature_names=splits.feature_names,
        num_samples=100,
        random_state=42
    )

    assert lime_res["explainer"] == "LIME_LocalSurrogate"
    assert "prediction" in lime_res
    assert len(lime_res["feature_contributions"]) == len(splits.feature_names)
    assert "weight" in lime_res["feature_contributions"][0]


def test_transformer_attention_visualizer():
    """Verify multi-head attention map extraction from TimeSeriesTransformerNN."""
    seq_splits = load_sensor_timeseries_dataset(num_samples=50, seq_len=10, num_features=6)
    model = TimeSeriesTransformerNN(in_features=6, d_model=32, nhead=4, num_layers=2)

    sample_seq = torch.tensor(seq_splits.X_val[0:1], dtype=torch.float32)
    attn_res = attention_visualizer.extract_attention_maps(model, sample_seq)

    assert attn_res["model_type"] == "TimeSeriesTransformer"
    assert attn_res["num_heads"] == 4
    assert attn_res["sequence_length"] == 10
    assert len(attn_res["averaged_attention_matrix"]) == 10
    assert len(attn_res["averaged_attention_matrix"][0]) == 10
    assert len(attn_res["temporal_importance"]) == 10
    assert 0 <= attn_res["top_attended_timestep"] < 10


def test_explainability_report_generator_and_html_export():
    """Verify comprehensive trust report synthesis and HTML export."""
    splits = load_starter_tabular_dataset()
    model = XGBoostModel(n_estimators=30, max_depth=3, random_state=42)
    model.fit(splits.X_train, splits.y_train)

    report = report_generator.generate_report(
        model=model,
        model_name="Wisconsin-XGBoost-V1",
        X_val=splits.X_val,
        y_val=splits.y_val,
        feature_names=splits.feature_names,
        security_events_count=0
    )

    assert report["model_name"] == "Wisconsin-XGBoost-V1"
    assert report["evaluation_metrics"]["accuracy"] > 0.85
    assert report["global_feature_importance"] is not None
    assert report["trust_and_governance"]["privacy_preserving_mode"] is not None

    # Export to HTML
    html = report_generator.export_report_to_html(report)
    assert "<!DOCTYPE html>" in html
    assert "Wisconsin-XGBoost-V1" in html
    assert "Privacy Compliant" in html
