"""
[IMPLEMENTED] Client database repository.
Manages federated client registrations, capability updates, and heartbeats.
"""
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.database.models_orm import Client


class ClientRepository:
    @staticmethod
    def register_or_update(
        db: Session,
        client_id: str,
        name: str,
        client_ip: str | None = None,
        capabilities: dict | None = None
    ) -> Client:
        """Register a new client or update existing client info."""
        client = db.query(Client).filter(Client.id == client_id).first()
        now = datetime.now(UTC)
        if not client:
            client = Client(
                id=client_id,
                name=name,
                client_ip=client_ip,
                status="online",
                capabilities=capabilities or {},
                last_heartbeat=now
            )
            db.add(client)
        else:
            client.name = name
            if client_ip:
                client.client_ip = client_ip
            if capabilities is not None:
                client.capabilities = capabilities
            client.status = "online"
            client.last_heartbeat = now

        db.commit()
        db.refresh(client)
        return client

    @staticmethod
    def get_by_id(db: Session, client_id: str) -> Client | None:
        """Fetch client by ID."""
        return db.query(Client).filter(Client.id == client_id).first()

    @staticmethod
    def list_online(db: Session) -> list[Client]:
        """List all online clients."""
        return db.query(Client).filter(Client.status == "online").all()

    @staticmethod
    def update_status(db: Session, client_id: str, status: str) -> Client | None:
        """Update client status."""
        client = db.query(Client).filter(Client.id == client_id).first()
        if client:
            client.status = status
            client.last_heartbeat = datetime.now(UTC)
            db.commit()
            db.refresh(client)
        return client


client_repo = ClientRepository()
