"""
[IMPLEMENTED] Unit and integration tests for Privacy & Security subsystems:
Differential Privacy Mechanism, Privacy Budget Accountant, Byzantine Threat Detector, and Security Audit Logger.
"""
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.connection import Base
from backend.security.audit_logger import audit_logger
from backend.security.dp_mechanism import (
    add_gaussian_noise,
    add_laplace_noise,
    apply_differential_privacy,
    clip_parameter_tensors,
    compute_l2_norm,
)
from backend.security.privacy_tracker import PrivacyBudgetTracker
from backend.security.threat_detector import ThreatDetector

# Test in-memory database
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


def test_differential_privacy_clipping_and_noise():
    """Verify DP L2-norm clipping and Gaussian/Laplace noise mechanisms."""
    params = [
        np.array([[2.0, 4.0], [4.0, 2.0]], dtype=np.float32),
        np.array([10.0, 10.0], dtype=np.float32)
    ]
    initial_norm = compute_l2_norm(params)
    assert initial_norm > 10.0

    # 1. Test Clipping
    clip_threshold = 2.5
    clipped_params, orig_norm = clip_parameter_tensors(params, clip_norm=clip_threshold)
    clipped_norm = compute_l2_norm(clipped_params)
    assert np.isclose(orig_norm, initial_norm)
    assert np.isclose(clipped_norm, clip_threshold, atol=1e-4)

    # 2. Test Gaussian Noise
    noisy_g, sigma = add_gaussian_noise(clipped_params, clip_norm=clip_threshold, epsilon=1.0, delta=1e-5, random_state=42)
    assert sigma > 0.0
    assert not np.array_equal(clipped_params[0], noisy_g[0])

    # 3. Test Laplace Noise
    _noisy_l, b = add_laplace_noise(clipped_params, clip_norm=clip_threshold, epsilon=1.0, random_state=42)
    assert b > 0.0

    # 4. Unified DP entrypoint
    dp_report = apply_differential_privacy(params, clip_norm=2.0, epsilon=0.5, delta=1e-5)
    assert dp_report["mechanism"] == "gaussian"
    assert dp_report["clipped_norm"] <= 2.0001
    assert len(dp_report["parameters"]) == 2


def test_privacy_budget_tracker_accounting():
    """Verify cumulative privacy expenditure and budget exhaustion."""
    tracker = PrivacyBudgetTracker(default_max_epsilon=5.0)

    # Spend 1.0 epsilon in round 1
    res1 = tracker.spend_budget(client_id="client_alpha", step_epsilon=1.0)
    assert res1["total_spent_epsilon"] == 1.0
    assert not res1["is_exhausted"]
    assert res1["remaining_budget"] == 4.0

    # Spend another 1.0 in round 2
    res2 = tracker.spend_budget(client_id="client_alpha", step_epsilon=1.0)
    assert res2["total_spent_epsilon"] > 1.0
    assert res2["rounds_participated"] == 2

    # Spend large budget to trigger exhaustion
    res3 = tracker.spend_budget(client_id="client_alpha", step_epsilon=4.0)
    assert res3["is_exhausted"]
    assert res3["remaining_budget"] == 0.0

    # Query status
    status = tracker.get_client_status("client_alpha")
    assert status["is_exhausted"]


def test_threat_detector_byzantine_filtering():
    """Verify threat detector flags gradient explosion and sign-flipping adversaries."""
    detector = ThreatDetector(min_cosine_similarity=-0.1, max_norm_multiplier=3.0)

    # 3 Benign updates pointing in positive direction
    benign_params = [
        ("client_1", [np.array([1.0, 1.2, 0.9])]),
        ("client_2", [np.array([1.1, 0.9, 1.0])]),
        ("client_3", [np.array([0.9, 1.0, 1.1])])
    ]

    # 1 Gradient explosion adversary
    explosion_params = ("adv_explosion", [np.array([50.0, 60.0, 55.0])])

    # 1 Sign-flipping adversary
    sign_flip_params = ("adv_signflip", [np.array([-1.0, -1.2, -0.9])])

    all_submissions = benign_params + [explosion_params, sign_flip_params]

    inspection = detector.inspect_client_updates(all_submissions)

    assert set(inspection.accepted_client_ids) == {"client_1", "client_2", "client_3"}
    assert "adv_explosion" in inspection.rejected_client_ids
    assert "adv_signflip" in inspection.rejected_client_ids
    assert len(inspection.anomaly_reports) == 2


def test_security_audit_logger_persistence():
    """Verify security events are persisted to database."""
    db = TestingSessionLocal()

    event = audit_logger.log_event(
        db=db,
        event_type="POISONING_ATTEMPT",
        severity="HIGH",
        client_id="malicious_node_9",
        ip_address="192.168.1.105",
        details={"reason": "sign_flipping_cosine_anomaly", "cosine_sim": -0.85}
    )

    assert event.id is not None
    assert event.event_type == "POISONING_ATTEMPT"
    assert event.severity == "HIGH"

    # Query recent events
    recent = audit_logger.get_recent_events(db=db, severity="HIGH")
    assert len(recent) == 1
    assert recent[0].client_id == "malicious_node_9"
    db.close()
