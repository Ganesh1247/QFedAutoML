"""
[IMPLEMENTED] CLI tool and simulation script to spin up N simulated edge clients with local data partitions.
"""
import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.connection import SessionLocal, init_db
from backend.federated.round_manager import round_manager
from backend.monitoring.logger import get_logger

logger = get_logger("ClientSimulator")


def main():
    parser = argparse.ArgumentParser(description="QFedAutoML Simulated Client Runner")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of simulated edge nodes")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--partition-mode", type=str, default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID concentration parameter")
    args = parser.parse_args()

    logger.info(f"Initializing database and dataset for {args.num_clients} simulated clients...")
    init_db()
    db = SessionLocal()

    try:
        logger.info(f"Starting Federated Learning simulation: {args.rounds} rounds, mode={args.partition_mode}")
        exp, results = round_manager.start_training_run(
            db=db,
            name=f"FL-CLI-Run-{args.num_clients}-clients",
            num_clients=args.num_clients,
            num_rounds=args.rounds,
            local_epochs=args.epochs,
            partition_mode=args.partition_mode,
            dirichlet_alpha=args.alpha
        )

        logger.info("==================================================")
        logger.info(f"Federated Simulation Complete! Experiment ID: {exp.id}")
        logger.info(f"Final Validation Accuracy: {results.get('final_val_accuracy', 0.0):.4f}")
        logger.info(f"Total Network Communication: {results.get('total_comm_mb', 0.0):.4f} MB")
        logger.info("==================================================")
    finally:
        db.close()


if __name__ == "__main__":
    main()
