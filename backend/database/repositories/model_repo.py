"""
[IMPLEMENTED] Model version registry repository.
Manages versioned global model checkpoints and production status.
"""
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.database.models_orm import ModelVersion


class ModelRepository:
    @staticmethod
    def register(
        db: Session,
        model_name: str,
        version: str,
        architecture_type: str,
        model_binary_path: str | None = None,
        hyperparameters: dict | None = None,
        validation_metrics: dict | None = None,
        is_production: bool = False
    ) -> ModelVersion:
        """Register a new model version."""
        mv = ModelVersion(
            model_name=model_name,
            version=version,
            architecture_type=architecture_type,
            model_binary_path=model_binary_path,
            hyperparameters=hyperparameters or {},
            validation_metrics=validation_metrics or {},
            is_production=is_production,
            created_at=datetime.now(UTC)
        )
        db.add(mv)
        db.commit()
        db.refresh(mv)
        return mv

    @staticmethod
    def get_latest_production(db: Session, model_name: str) -> ModelVersion | None:
        """Retrieve current production version for a model."""
        return db.query(ModelVersion).filter(
            ModelVersion.model_name == model_name,
            ModelVersion.is_production.is_(True)
        ).order_by(ModelVersion.id.desc()).first()

    @staticmethod
    def list_versions(db: Session, model_name: str) -> list[ModelVersion]:
        """List all versions of a model."""
        return db.query(ModelVersion).filter(
            ModelVersion.model_name == model_name
        ).order_by(ModelVersion.id.desc()).all()


model_repo = ModelRepository()
