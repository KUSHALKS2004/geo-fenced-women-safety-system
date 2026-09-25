from datetime import datetime, timezone
from pathlib import Path
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_current_user
from database.models import User


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/sos",
    tags=["Emergency Assistance"]
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ALERTS_DIR = BASE_DIR / "alerts"

ALERTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOS_FILE = ALERTS_DIR / "sos_events.json"


# ============================================================
# REQUEST MODEL
# ============================================================

class SOSRequest(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="User latitude"
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="User longitude"
    )

    risk_band: str = Field(
        ...,
        description="Current risk band"
    )

    risk_share: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Model-based predicted women-crime share"
    )

    state: str | None = None

    district: str | None = None


# ============================================================
# LOAD EXISTING SOS EVENTS
# ============================================================

def load_sos_events():

    if not SOS_FILE.exists():
        return []

    try:

        with open(
            SOS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except (json.JSONDecodeError, OSError):

        return []


# ============================================================
# SAVE SOS EVENTS
# ============================================================

def save_sos_events(events):

    with open(
        SOS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            events,
            file,
            indent=4
        )


# ============================================================
# TRIGGER SOS
# ============================================================

@router.post("/trigger")
def trigger_sos(
    request: SOSRequest,
    current_user: User = Depends(get_current_user)
):

    try:

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        event = {

            "event_id": (
                f"SOS-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            ),

            "timestamp_utc": timestamp,

            "user_id": current_user.id,

            "user_name": current_user.full_name,

            "user_email": current_user.email,

            "latitude": request.latitude,

            "longitude": request.longitude,

            "state": request.state,

            "district": request.district,

            "risk_band": request.risk_band,

            "risk_share": request.risk_share,

            "status": "SOS_EVENT_RECORDED",

            "message": (
                "Emergency assistance event "
                "recorded successfully."
            )
        }

        events = load_sos_events()

        events.append(event)

        save_sos_events(events)

        return {

            "success": True,

            "event_id": event["event_id"],

            "status": event["status"],

            "timestamp_utc": timestamp,

            "latitude": request.latitude,

            "longitude": request.longitude,

            "state": request.state,

            "district": request.district,

            "risk_band": request.risk_band,

            "message": (
                "SOS event recorded. "
                "Use the emergency call or location-sharing "
                "options for immediate assistance."
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"SOS processing failed: {error}"
        )