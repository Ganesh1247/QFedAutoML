"""
[IMPLEMENTED] Federated Round and Experiment Orchestration Manager.
"""
from typing import Any

from sqlalchemy.orm import Session

from backend.automl.preprocessing import (
    load_sensor_timeseries_dataset,
    load_starter_tabular_dataset,
)
from backend.database.models_orm import Dataset, Experiment
from backend.database.repositories.client_repo import client_repo
from backend.database.repositories.training_repo import training_repo
from backend.federated.server import run_federated_simulation
from clients_simulation.data_partitioner import (
    partition_data_iid,
    partition_data_non_iid_dirichlet,
)


class FederatedRoundManager:
    """Manages the lifecycle of federated training runs."""

    @staticmethod
    def start_training_run(
        db: Session,
        name: str = "FL Federated Baseline",
        num_clients: int = 5,
        num_rounds: int = 5,
        local_epochs: int = 2,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        partition_mode: str = "non_iid",
        dirichlet_alpha: float = 0.5,
        model_architecture: str = "tabular_nn",
        dataset_type: str = "tabular",
        baseline_type: str = "federated_classical"
    ) -> tuple[Experiment, dict[str, Any]]:
        """
        Orchestrate an end-to-end federated training experiment:
        1. Load dataset (Tabular or Sequence) & create DB record
        2. Partition data among N simulated clients
        3. Register client nodes in DB
        4. Create Experiment record
        5. Run Flower simulation and persist round metrics
        """
        if dataset_type == "sequence" or model_architecture == "transformer":
            seq_splits = load_sensor_timeseries_dataset()
            dataset_name = "sensor_timeseries_seq"
            num_samples = seq_splits.num_samples
            num_features = seq_splits.num_features
            target_name = "activity_class"
            feature_names = [f"sensor_axis_{i}" for i in range(num_features)]
            X_data = seq_splits.X_train
            y_data = seq_splits.y_train
        else:
            splits = load_starter_tabular_dataset()
            dataset_name = "breast_cancer_tabular"
            num_samples = splits.num_samples
            num_features = splits.num_features
            target_name = splits.target_name
            feature_names = splits.feature_names
            X_data = splits.X_train
            y_data = splits.y_train

        # Check or create dataset record
        dataset_rec = db.query(Dataset).filter(Dataset.name == dataset_name).first()
        if not dataset_rec:
            dataset_rec = Dataset(
                name=dataset_name,
                dataset_type=dataset_type,
                num_samples=num_samples,
                num_features=num_features,
                feature_names=feature_names,
                target_name=target_name
            )
            db.add(dataset_rec)
            db.commit()
            db.refresh(dataset_rec)

        # Partition data
        if partition_mode == "iid":
            partitions = partition_data_iid(
                X_data, y_data, num_clients=num_clients
            )
        else:
            partitions = partition_data_non_iid_dirichlet(
                X_data, y_data, num_clients=num_clients, alpha=dirichlet_alpha
            )

        # Register clients in DB
        for part in partitions:
            client_repo.register_or_update(
                db=db,
                client_id=part.client_id,
                name=f"Edge Node {part.client_id.upper()}",
                capabilities={"samples": part.num_samples, "features": num_features, "model": model_architecture}
            )

        # Create experiment record
        exp = training_repo.create_experiment(
            db=db,
            name=name,
            baseline_type=baseline_type,
            dataset_id=dataset_rec.id,
            config={
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "local_epochs": local_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "partition_mode": partition_mode,
                "dirichlet_alpha": dirichlet_alpha,
                "model_architecture": model_architecture,
                "dataset_type": dataset_type
            }
        )

        # Run simulation
        results = run_federated_simulation(
            partitions=partitions,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_architecture=model_architecture,
            experiment_id=exp.id,
            db=db
        )

        return exp, results



round_manager = FederatedRoundManager()
