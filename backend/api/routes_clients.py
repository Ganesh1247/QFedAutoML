"""
[IMPLEMENTED] Edge Clients Management REST APIs.
Handles edge client device registration, heartbeat polling, and telemetry querying.
"""
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database.models_orm import Client
from backend.dependencies import get_db
from backend.security.privacy_tracker import privacy_tracker

router = APIRouter(prefix="/clients", tags=["Edge Clients"])


# --- Schemas ---
class ClientRegisterRequest(BaseModel):
    id: str = Field(..., description="Unique client hardware or UUID identifier")
    name: str = Field(..., description="Human-readable client node name")
    client_ip: str | None = "127.0.0.1"
    device_info: dict[str, Any] = Field(default_factory=dict, description="Hardware specs (CPU, GPU, RAM, OS)")
    data_samples_count: int = Field(default=0, ge=0)
    data_quality_score: float = Field(default=0.8, ge=0.0, le=1.0)


class ClientHeartbeatRequest(BaseModel):
    status: str = Field(default="online")
    data_quality_score: float | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)


class ClientResponse(BaseModel):
    id: str
    name: str
    status: str
    device_info: dict[str, Any]
    data_samples_count: int
    data_quality_score: float
    registered_at: datetime
    last_seen_at: datetime
    privacy_status: dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


# --- Endpoints ---
@router.post("/register", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
def register_client(
    req: ClientRegisterRequest,
    db: Session = Depends(get_db)
):
    """Register or update an edge client node in the federated network."""
    client = db.query(Client).filter(Client.id == req.id).first()
    now = datetime.now(UTC)
    capabilities_dict = dict(req.device_info)
    capabilities_dict["data_samples_count"] = req.data_samples_count

    if client:
        client.name = req.name
        client.client_ip = req.client_ip
        client.capabilities = capabilities_dict
        client.reliability_score = req.data_quality_score
        client.last_heartbeat = now
        client.status = "online"
    else:
        client = Client(
            id=req.id,
            name=req.name,
            client_ip=req.client_ip,
            capabilities=capabilities_dict,
            reliability_score=req.data_quality_score,
            status="online",
            created_at=now,
            last_heartbeat=now
        )
        db.add(client)

    db.commit()
    db.refresh(client)

    return {
        "id": client.id,
        "name": client.name,
        "status": client.status,
        "device_info": client.capabilities,
        "data_samples_count": int(client.capabilities.get("data_samples_count", 0)),
        "data_quality_score": client.reliability_score,
        "registered_at": client.created_at,
        "last_seen_at": client.last_heartbeat,
        "privacy_status": privacy_tracker.get_client_status(client.id)
    }


@router.post("/{client_id}/heartbeat", response_model=ClientResponse)
def client_heartbeat(
    client_id: str,
    req: ClientHeartbeatRequest,
    db: Session = Depends(get_db)
):
    """Update heartbeat timestamp and online telemetry for a client."""
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{client_id}' not found. Please register first."
        )

    client.last_heartbeat = datetime.now(UTC)
    client.status = req.status
    if req.data_quality_score is not None:
        client.reliability_score = req.data_quality_score

    db.commit()
    db.refresh(client)

    return {
        "id": client.id,
        "name": client.name,
        "status": client.status,
        "device_info": client.capabilities,
        "data_samples_count": int(client.capabilities.get("data_samples_count", 0)),
        "data_quality_score": client.reliability_score,
        "registered_at": client.created_at,
        "last_seen_at": client.last_heartbeat,
        "privacy_status": privacy_tracker.get_client_status(client.id)
    }


@router.get("", response_model=list[ClientResponse])
def list_clients(
    status_filter: str | None = None,
    db: Session = Depends(get_db)
):
    """Retrieve all edge client devices with their current status and privacy budget."""
    query = db.query(Client)
    if status_filter:
        query = query.filter(Client.status == status_filter.lower())

    clients = query.order_by(Client.last_heartbeat.desc()).all()
    results = []
    for c in clients:
        results.append({
            "id": c.id,
            "name": c.name,
            "status": c.status,
            "device_info": c.capabilities,
            "data_samples_count": int(c.capabilities.get("data_samples_count", 0)),
            "data_quality_score": c.reliability_score,
            "registered_at": c.created_at,
            "last_seen_at": c.last_heartbeat,
            "privacy_status": privacy_tracker.get_client_status(c.id)
        })
    return results


@router.get("/{client_id}", response_model=ClientResponse)
def get_client_detail(
    client_id: str,
    db: Session = Depends(get_db)
):
    """Get detailed telemetry and privacy budget expenditure for a client."""
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client '{client_id}' not found."
        )

    return {
        "id": client.id,
        "name": client.name,
        "status": client.status,
        "device_info": client.capabilities,
        "data_samples_count": int(client.capabilities.get("data_samples_count", 0)),
        "data_quality_score": client.reliability_score,
        "registered_at": client.created_at,
        "last_seen_at": client.last_heartbeat,
        "privacy_status": privacy_tracker.get_client_status(client.id)
    }
