"""
[IMPLEMENTED] Federated telemetry and communication metrics database logger.
"""
from typing import Any

from sqlalchemy.orm import Session

from backend.database.models_orm import Metric, TrainingRound


def log_federated_round(
    db: Session,
    experiment_id: int,
    round_telemetry: dict[str, Any],
    selected_client_ids: list[str] | None = None
) -> TrainingRound:
    """
    Persist round execution details, accuracy, loss, and communication costs into the database.
    """
    round_num = int(round_telemetry.get("round", 1))
    loss = round_telemetry.get("val_loss", round_telemetry.get("train_loss"))
    acc = round_telemetry.get("val_accuracy", round_telemetry.get("train_accuracy"))

    # Check if round record exists
    t_round = db.query(TrainingRound).filter(
        TrainingRound.experiment_id == experiment_id,
        TrainingRound.round_number == round_num
    ).first()

    if not t_round:
        t_round = TrainingRound(
            experiment_id=experiment_id,
            round_number=round_num,
            status="completed",
            selected_client_ids=selected_client_ids or [],
            loss=float(loss) if loss is not None else None,
            accuracy=float(acc) if acc is not None else None,
            round_metrics=round_telemetry
        )
        db.add(t_round)
    else:
        t_round.loss = float(loss) if loss is not None else None
        t_round.accuracy = float(acc) if acc is not None else None
        t_round.round_metrics = round_telemetry
        t_round.status = "completed"

    db.commit()
    db.refresh(t_round)

    # Log scalar metrics into time-series table
    metric_fields = [
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_f1",
        "total_comm_mb",
        "duration_seconds"
    ]
    for field in metric_fields:
        if field in round_telemetry and round_telemetry[field] is not None:
            db_metric = Metric(
                experiment_id=experiment_id,
                round_id=t_round.id,
                metric_name=f"fl_{field}",
                metric_value=float(round_telemetry[field]),
                step=round_num
            )
            db.add(db_metric)

    db.commit()
    return t_round
