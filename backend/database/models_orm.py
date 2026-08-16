"""
[IMPLEMENTED] SQLAlchemy ORM Schema Definitions:
users, clients, datasets, experiments, training_rounds, model_versions,
quantum_jobs, metrics, predictions, security_events.
"""
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False
    )


class Client(Base):
    __tablename__ = "clients"

    id = Column(String(64), primary_key=True, index=True)  # Client UUID or node ID
    name = Column(String(128), nullable=False)
    client_ip = Column(String(64), nullable=True)
    status = Column(String(32), default="registered", index=True, nullable=False)  # registered, online, training, offline
    capabilities = Column(JSON, default=dict, nullable=False)  # {"cpu_cores": 4, "ram_gb": 16, "device": "cuda"}
    reliability_score = Column(Float, default=1.0, nullable=False)
    last_heartbeat = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    security_events = relationship("SecurityEvent", back_populates="client")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), unique=True, index=True, nullable=False)
    dataset_type = Column(String(32), default="tabular", nullable=False)  # tabular, sequence
    description = Column(Text, nullable=True)
    num_samples = Column(Integer, default=0, nullable=False)
    num_features = Column(Integer, default=0, nullable=False)
    feature_names = Column(JSON, default=list, nullable=False)
    target_name = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    experiments = relationship("Experiment", back_populates="dataset")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), index=True, nullable=False)
    description = Column(Text, nullable=True)
    baseline_type = Column(
        String(64),
        nullable=False
    )  # centralized_classical, federated_classical, federated_automl, federated_quantum
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    status = Column(String(32), default="created", index=True, nullable=False)  # created, running, completed, failed
    config = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    dataset = relationship("Dataset", back_populates="experiments")
    training_rounds = relationship("TrainingRound", back_populates="experiment", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="experiment", cascade="all, delete-orphan")


class TrainingRound(Base):
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False, index=True)
    round_number = Column(Integer, nullable=False)
    status = Column(String(32), default="pending", nullable=False)  # pending, in_progress, completed, failed
    selected_client_ids = Column(JSON, default=list, nullable=False)
    aggregation_strategy = Column(String(64), default="fedavg", nullable=False)
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    round_metrics = Column(JSON, default=dict, nullable=False)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    experiment = relationship("Experiment", back_populates="training_rounds")
    metrics = relationship("Metric", back_populates="training_round")

    __table_args__ = (
        Index("idx_experiment_round", "experiment_id", "round_number", unique=True),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(128), index=True, nullable=False)
    version = Column(String(32), nullable=False)  # e.g. "v1.0.0"
    architecture_type = Column(String(64), nullable=False)  # xgboost, mlp, transformer, logistic_regression
    model_binary_path = Column(String(512), nullable=True)
    hyperparameters = Column(JSON, default=dict, nullable=False)
    validation_metrics = Column(JSON, default=dict, nullable=False)  # {"accuracy": 0.92, "f1": 0.91, "roc_auc": 0.96}
    is_production = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    predictions = relationship("Prediction", back_populates="model_version")

    __table_args__ = (
        Index("idx_model_name_version", "model_name", "version", unique=True),
    )


class QuantumJob(Base):
    __tablename__ = "quantum_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String(64), index=True, nullable=False)  # feature_selection, client_selection, hpo
    status = Column(String(32), default="queued", index=True, nullable=False)  # queued, running, completed, failed
    backend_used = Column(String(64), default="qiskit_aer", nullable=False)  # qiskit_aer, pennylane_default, classical_fallback
    num_qubits = Column(Integer, default=0, nullable=False)
    circuit_depth = Column(Integer, default=0, nullable=False)
    objective_value = Column(Float, nullable=True)
    classical_objective_value = Column(Float, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    classical_time_ms = Column(Float, nullable=True)
    parameters = Column(JSON, default=dict, nullable=False)
    result = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    metrics = relationship("Metric", back_populates="quantum_job")


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True, index=True)
    round_id = Column(Integer, ForeignKey("training_rounds.id"), nullable=True, index=True)
    job_id = Column(Integer, ForeignKey("quantum_jobs.id"), nullable=True, index=True)
    metric_name = Column(String(64), index=True, nullable=False)
    metric_value = Column(Float, nullable=False)
    step = Column(Integer, default=0, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    experiment = relationship("Experiment", back_populates="metrics")
    training_round = relationship("TrainingRound", back_populates="metrics")
    quantum_job = relationship("QuantumJob", back_populates="metrics")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False, index=True)
    input_data = Column(JSON, nullable=False)
    prediction_output = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    model_version = relationship("ModelVersion", back_populates="predictions")


class SecurityEvent(Base):
    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(64), index=True, nullable=False)  # auth_failure, poisoning_attempt, byzantine_anomaly, dp_budget_exceeded
    severity = Column(String(16), default="info", nullable=False)  # info, warning, critical
    client_id = Column(String(64), ForeignKey("clients.id"), nullable=True, index=True)
    details = Column(JSON, default=dict, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    client = relationship("Client", back_populates="security_events")
