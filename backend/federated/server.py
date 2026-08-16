"""
[IMPLEMENTED] Flower-compatible Federated Learning Server orchestrator and simulation runner.
Decentralized aggregation executes using FedAvgWithTelemetry and Flower NumPyClients.
"""
from typing import Any

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from sqlalchemy.orm import Session

from backend.database.models_orm import Experiment
from backend.evaluation.federated_metrics import log_federated_round
from backend.federated.client import (
    FlowerTabularClient,
    TabularPyTorchNN,
    get_model_parameters,
)
from backend.federated.strategies.fedavg import FedAvgWithTelemetry
from backend.models.transformer_model import TimeSeriesTransformerNN
from clients_simulation.data_partitioner import ClientDataPartition


class SimulatedClientProxy(ClientProxy):
    """Proxy representing a simulated edge node in local execution."""

    def __init__(self, cid: str, client: FlowerTabularClient):
        super().__init__(cid)
        self.client = client

    def get_properties(self, ins, timeout, group_id):
        return None

    def get_parameters(self, ins, timeout, group_id):
        params = self.client.get_parameters({})
        return ndarrays_to_parameters(params)

    def fit(self, ins: FitIns, timeout, group_id) -> FitRes:
        params = parameters_to_ndarrays(ins.parameters)
        updated_params, num_samples, metrics = self.client.fit(params, ins.config)
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=num_samples,
            metrics=metrics
        )

    def evaluate(self, ins: EvaluateIns, timeout, group_id) -> EvaluateRes:
        params = parameters_to_ndarrays(ins.parameters)
        loss, num_samples, metrics = self.client.evaluate(params, ins.config)
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=num_samples,
            metrics=metrics
        )

    def reconnect(self, ins, timeout, group_id):
        pass


def run_federated_simulation(
    partitions: list[ClientDataPartition],
    num_rounds: int = 5,
    local_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    fraction_fit: float = 1.0,
    model_architecture: str = "tabular_nn",
    experiment_id: int | None = None,
    db: Session | None = None
) -> dict[str, Any]:
    """
    Run decentralized Federated Learning simulation across edge client partitions.
    Preserves strict data locality (raw samples remain inside each client's partition).
    """
    num_clients = len(partitions)
    in_features = partitions[0].X_train.shape[1] if partitions[0].X_train.ndim == 2 else partitions[0].X_train.shape[2]

    # Initialize initial global model parameters
    if model_architecture == "transformer":
        init_model = TimeSeriesTransformerNN(in_features=in_features, d_model=32, nhead=4, num_layers=2)
    else:
        init_model = TabularPyTorchNN(in_features=in_features, hidden_dim=64)

    current_parameters = ndarrays_to_parameters(get_model_parameters(init_model))

    # Configure custom strategy with communication telemetry
    strategy = FedAvgWithTelemetry(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=max(2, int(num_clients * fraction_fit)),
        min_evaluate_clients=max(2, int(num_clients * fraction_fit)),
        min_available_clients=num_clients,
        initial_parameters=current_parameters,
    )

    # Instantiate simulated edge clients
    clients = [
        FlowerTabularClient(
            client_id=p.client_id,
            partition=p,
            in_features=in_features,
            model_architecture=model_architecture,
            device="cpu"
        )
        for p in partitions
    ]
    proxies = [SimulatedClientProxy(cid=f"{i}", client=c) for i, c in enumerate(clients)]

    # Multi-round federated training loop
    for server_round in range(1, num_rounds + 1):
        fit_config = {
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "round": server_round
        }
        fit_ins = FitIns(parameters=current_parameters, config=fit_config)

        # 1. Dispatch fit to clients
        fit_results: list[tuple[ClientProxy, FitRes]] = []
        for proxy in proxies:
            res = proxy.fit(fit_ins, timeout=None, group_id=None)
            fit_results.append((proxy, res))

        # 2. Server Aggregation via FedAvgWithTelemetry
        aggregated_params, _ = strategy.aggregate_fit(
            server_round=server_round,
            results=fit_results,
            failures=[]
        )
        if aggregated_params is not None:
            current_parameters = aggregated_params

        # 3. Federated Evaluation across clients
        eval_ins = EvaluateIns(parameters=current_parameters, config={"round": server_round})
        eval_results: list[tuple[ClientProxy, EvaluateRes]] = []
        for proxy in proxies:
            res = proxy.evaluate(eval_ins, timeout=None, group_id=None)
            eval_results.append((proxy, res))

        strategy.aggregate_evaluate(
            server_round=server_round,
            results=eval_results,
            failures=[]
        )

    # Extract round telemetry history
    round_history = strategy.round_history

    # Log to database if DB session and experiment_id are provided
    if db is not None and experiment_id is not None:
        client_ids = [p.client_id for p in partitions]
        for round_data in round_history:
            log_federated_round(
                db=db,
                experiment_id=experiment_id,
                round_telemetry=round_data,
                selected_client_ids=client_ids
            )
        # Update experiment status to completed
        exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if exp:
            exp.status = "completed"
            db.commit()

    final_round = round_history[-1] if round_history else {}
    total_comm_mb = sum(r.get("total_comm_mb", 0.0) for r in round_history)

    return {
        "status": "completed",
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "final_train_loss": final_round.get("train_loss", 0.0),
        "final_train_accuracy": final_round.get("train_accuracy", 0.0),
        "final_val_accuracy": final_round.get("val_accuracy", final_round.get("train_accuracy", 0.0)),
        "final_val_f1": final_round.get("val_f1", 0.0),
        "total_comm_mb": total_comm_mb,
        "round_history": round_history
    }
