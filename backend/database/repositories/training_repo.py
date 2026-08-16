"""
[IMPLEMENTED] Training repository.
Tracks federated experiments, training rounds, and aggregated metrics.
"""
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.database.models_orm import Experiment, Metric, TrainingRound


class TrainingRepository:
    @staticmethod
    def create_experiment(
        db: Session,
        name: str,
        baseline_type: str,
        description: str | None = None,
        dataset_id: int | None = None,
        config: dict | None = None
    ) -> Experiment:
        """Create a new experiment record."""
        exp = Experiment(
            name=name,
            baseline_type=baseline_type,
            description=description,
            dataset_id=dataset_id,
            config=config or {},
            status="created"
        )
        db.add(exp)
        db.commit()
        db.refresh(exp)
        return exp

    @staticmethod
    def get_experiment(db: Session, experiment_id: int) -> Experiment | None:
        """Fetch experiment by ID."""
        return db.query(Experiment).filter(Experiment.id == experiment_id).first()

    @staticmethod
    def create_round(
        db: Session,
        experiment_id: int,
        round_number: int,
        selected_client_ids: list[str],
        aggregation_strategy: str = "fedavg"
    ) -> TrainingRound:
        """Create a new federated training round record."""
        t_round = TrainingRound(
            experiment_id=experiment_id,
            round_number=round_number,
            status="in_progress",
            selected_client_ids=selected_client_ids,
            aggregation_strategy=aggregation_strategy,
            started_at=datetime.now(UTC)
        )
        db.add(t_round)
        db.commit()
        db.refresh(t_round)
        return t_round

    @staticmethod
    def finish_round(
        db: Session,
        round_id: int,
        loss: float | None = None,
        accuracy: float | None = None,
        metrics: dict | None = None,
        status: str = "completed"
    ) -> TrainingRound | None:
        """Complete a training round with metrics."""
        t_round = db.query(TrainingRound).filter(TrainingRound.id == round_id).first()
        if t_round:
            t_round.loss = loss
            t_round.accuracy = accuracy
            t_round.round_metrics = metrics or {}
            t_round.status = status
            t_round.completed_at = datetime.now(UTC)
            db.commit()
            db.refresh(t_round)
        return t_round

    @staticmethod
    def log_metric(
        db: Session,
        metric_name: str,
        metric_value: float,
        step: int = 0,
        experiment_id: int | None = None,
        round_id: int | None = None
    ) -> Metric:
        """Log a telemetry metric entry."""
        metric = Metric(
            metric_name=metric_name,
            metric_value=metric_value,
            step=step,
            experiment_id=experiment_id,
            round_id=round_id
        )
        db.add(metric)
        db.commit()
        db.refresh(metric)
        return metric


training_repo = TrainingRepository()
